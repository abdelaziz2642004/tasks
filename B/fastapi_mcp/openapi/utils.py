import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Maximum depth for reference resolution to prevent stack overflow
MAX_REFERENCE_DEPTH = 50

# Maximum number of references to resolve to prevent infinite loops
MAX_REFERENCE_COUNT = 1000


@dataclass
class ReferenceResolutionContext:
    """Context for tracking reference resolution state."""

    # Set of reference paths currently being resolved (for circular detection)
    resolution_stack: Set[str] = field(default_factory=set)
    # Cache of already resolved references
    resolved_cache: Dict[str, Any] = field(default_factory=dict)
    # Current resolution depth
    depth: int = 0
    # Total number of references resolved
    reference_count: int = 0
    # Detected circular references
    circular_refs: Set[str] = field(default_factory=set)
    # Warnings for problematic patterns
    warnings: List[str] = field(default_factory=list)


@dataclass
class ReferenceResolutionResult:
    """Result of reference resolution including any warnings."""

    schema: Dict[str, Any]
    circular_refs: Set[str]
    warnings: List[str]
    reference_count: int


def get_single_param_type_from_schema(param_schema: Dict[str, Any]) -> str:
    """
    Get the type of a parameter from the schema.
    If the schema is a union type, return the first type.
    """
    if "anyOf" in param_schema:
        types = {schema.get("type") for schema in param_schema["anyOf"] if schema.get("type")}
        if "null" in types:
            types.remove("null")
        if types:
            return next(iter(types))
        return "string"
    return param_schema.get("type", "string")


def _resolve_json_pointer(reference_schema: Dict[str, Any], ref_path: str) -> Optional[Dict[str, Any]]:
    """
    Resolve a JSON pointer reference path to the actual schema.

    Args:
        reference_schema: The complete schema containing the referenced definition
        ref_path: The reference path (e.g., "#/components/schemas/ModelName")

    Returns:
        The resolved schema or None if not found
    """
    if not ref_path.startswith("#/"):
        # External references are not supported
        return None

    # Remove the leading "#/" and split by "/"
    path_parts = ref_path[2:].split("/")

    current = reference_schema
    for part in path_parts:
        # Handle URL-encoded characters in JSON pointer (RFC 6901)
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None

    return current if isinstance(current, dict) else None


def _resolve_schema_references_internal(
    schema_part: Any,
    reference_schema: Dict[str, Any],
    context: ReferenceResolutionContext,
) -> Any:
    """
    Internal recursive function to resolve schema references.

    Args:
        schema_part: The part of the schema being processed
        reference_schema: The complete schema used to resolve references from
        context: The resolution context for tracking state

    Returns:
        The schema with references resolved
    """
    # Handle non-dict types
    if not isinstance(schema_part, dict):
        if isinstance(schema_part, list):
            return [
                _resolve_schema_references_internal(item, reference_schema, context)
                for item in schema_part
            ]
        return schema_part

    # Check depth limit
    if context.depth > MAX_REFERENCE_DEPTH:
        warning = f"Maximum reference depth ({MAX_REFERENCE_DEPTH}) exceeded"
        if warning not in context.warnings:
            context.warnings.append(warning)
            logger.warning(warning)
        return schema_part

    # Check reference count limit
    if context.reference_count > MAX_REFERENCE_COUNT:
        warning = f"Maximum reference count ({MAX_REFERENCE_COUNT}) exceeded"
        if warning not in context.warnings:
            context.warnings.append(warning)
            logger.warning(warning)
        return schema_part

    # Make a deep copy to avoid modifying the input schema
    result = {}

    # Handle $ref directly in the schema
    if "$ref" in schema_part:
        ref_path = schema_part["$ref"]
        context.reference_count += 1

        # Check for circular reference
        if ref_path in context.resolution_stack:
            context.circular_refs.add(ref_path)
            logger.debug(f"Circular reference detected: {ref_path}")
            # Return a placeholder for circular references
            # Keep the $ref but mark it as circular
            return {"$ref": ref_path, "_circular": True}

        # Check cache first
        if ref_path in context.resolved_cache:
            # Return a deep copy of cached result to avoid mutation issues
            return copy.deepcopy(context.resolved_cache[ref_path])

        # Resolve the reference
        resolved = _resolve_json_pointer(reference_schema, ref_path)
        if resolved is not None:
            # Add to resolution stack before recursing
            context.resolution_stack.add(ref_path)
            context.depth += 1

            try:
                # Recursively resolve any nested references in the resolved schema
                resolved_copy = _resolve_schema_references_internal(
                    resolved, reference_schema, context
                )

                # Merge any additional properties from the original schema (excluding $ref)
                for key, value in schema_part.items():
                    if key != "$ref":
                        if isinstance(resolved_copy, dict):
                            resolved_copy[key] = _resolve_schema_references_internal(
                                value, reference_schema, context
                            )

                # Cache the result
                context.resolved_cache[ref_path] = copy.deepcopy(resolved_copy)

                return resolved_copy
            finally:
                # Remove from resolution stack
                context.resolution_stack.discard(ref_path)
                context.depth -= 1
        else:
            # Reference could not be resolved - keep original
            warning = f"Could not resolve reference: {ref_path}"
            if warning not in context.warnings:
                context.warnings.append(warning)
                logger.warning(warning)
            return copy.deepcopy(schema_part)

    # No $ref - process all keys recursively
    context.depth += 1
    try:
        for key, value in schema_part.items():
            if isinstance(value, dict):
                result[key] = _resolve_schema_references_internal(
                    value, reference_schema, context
                )
            elif isinstance(value, list):
                result[key] = [
                    _resolve_schema_references_internal(item, reference_schema, context)
                    if isinstance(item, (dict, list))
                    else item
                    for item in value
                ]
            else:
                result[key] = value
    finally:
        context.depth -= 1

    return result


def resolve_schema_references(
    schema_part: Dict[str, Any],
    reference_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve schema references in OpenAPI schemas.

    This function handles:
    - Standard OpenAPI references (#/components/schemas/ModelName)
    - Other component references (#/components/parameters/ParamName, etc.)
    - Circular references (detected and handled gracefully)
    - Deeply nested references (with depth limiting)
    - Caching for efficiency

    Args:
        schema_part: The part of the schema being processed that may contain references
        reference_schema: The complete schema used to resolve references from

    Returns:
        The schema with references resolved
    """
    context = ReferenceResolutionContext()
    result = _resolve_schema_references_internal(schema_part, reference_schema, context)

    if context.circular_refs:
        logger.info(f"Resolved schema with {len(context.circular_refs)} circular reference(s)")

    if context.warnings:
        logger.info(f"Reference resolution completed with {len(context.warnings)} warning(s)")

    return result if isinstance(result, dict) else schema_part


def resolve_schema_references_with_details(
    schema_part: Dict[str, Any],
    reference_schema: Dict[str, Any],
) -> ReferenceResolutionResult:
    """
    Resolve schema references and return detailed information about the resolution.

    This is useful for debugging and for detecting problematic reference patterns.

    Args:
        schema_part: The part of the schema being processed that may contain references
        reference_schema: The complete schema used to resolve references from

    Returns:
        A ReferenceResolutionResult containing the resolved schema and metadata
    """
    context = ReferenceResolutionContext()
    result = _resolve_schema_references_internal(schema_part, reference_schema, context)

    return ReferenceResolutionResult(
        schema=result if isinstance(result, dict) else schema_part,
        circular_refs=context.circular_refs,
        warnings=context.warnings,
        reference_count=context.reference_count,
    )


def clean_schema_for_display(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up a schema for display by removing internal fields.

    Args:
        schema: The schema to clean

    Returns:
        The cleaned schema
    """
    # Make a copy to avoid modifying the input schema
    schema = schema.copy()

    # Remove common internal fields that are not helpful for LLMs
    fields_to_remove = [
        "allOf",
        "anyOf",
        "oneOf",
        "nullable",
        "discriminator",
        "readOnly",
        "writeOnly",
        "xml",
        "externalDocs",
    ]
    for field in fields_to_remove:
        if field in schema:
            schema.pop(field)

    # Process nested properties
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict):
                schema["properties"][prop_name] = clean_schema_for_display(prop_schema)

    # Process array items
    if "type" in schema and schema["type"] == "array" and "items" in schema:
        if isinstance(schema["items"], dict):
            schema["items"] = clean_schema_for_display(schema["items"])

    return schema


def generate_example_from_schema(schema: Dict[str, Any]) -> Any:
    """
    Generate a simple example response from a JSON schema.

    Args:
        schema: The JSON schema to generate an example from

    Returns:
        An example object based on the schema
    """
    if not schema or not isinstance(schema, dict):
        return None

    # Handle different types
    schema_type = schema.get("type")

    if schema_type == "object":
        result = {}
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                # Generate an example for each property
                prop_example = generate_example_from_schema(prop_schema)
                if prop_example is not None:
                    result[prop_name] = prop_example
        return result

    elif schema_type == "array":
        if "items" in schema:
            # Generate a single example item
            item_example = generate_example_from_schema(schema["items"])
            if item_example is not None:
                return [item_example]
        return []

    elif schema_type == "string":
        # Check if there's a format
        format_type = schema.get("format")
        if format_type == "date-time":
            return "2023-01-01T00:00:00Z"
        elif format_type == "date":
            return "2023-01-01"
        elif format_type == "email":
            return "user@example.com"
        elif format_type == "uri":
            return "https://example.com"
        # Use title or property name if available
        return schema.get("title", "string")

    elif schema_type == "integer":
        return 1

    elif schema_type == "number":
        return 1.0

    elif schema_type == "boolean":
        return True

    elif schema_type == "null":
        return None

    # Default case
    return None
