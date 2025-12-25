import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
    # Cache of already resolved references (stores the resolved schema directly)
    resolved_cache: Dict[str, Any] = field(default_factory=dict)
    # Cache for schemas that need deep copy (only when they have nested refs)
    needs_deep_copy: Set[str] = field(default_factory=set)
    # Current resolution depth
    depth: int = 0
    # Total number of references resolved
    reference_count: int = 0
    # Cache hits for performance monitoring
    cache_hits: int = 0
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
    cache_hits: int = 0
    # Unresolved references found during validation
    unresolved_refs: Set[str] = field(default_factory=set)


class UnresolvedReferenceError(Exception):
    """Raised when unresolved references are found after resolution."""

    def __init__(self, unresolved_refs: Set[str], message: str = None):
        self.unresolved_refs = unresolved_refs
        if message is None:
            refs_list = ", ".join(sorted(unresolved_refs))
            message = (
                f"Found {len(unresolved_refs)} unresolved reference(s) after resolution: {refs_list}. "
                "This may indicate missing schema definitions or external references that cannot be resolved."
            )
        super().__init__(message)


# Schema metadata fields that should be preserved during reference resolution
SCHEMA_METADATA_FIELDS = {
    "title",
    "description",
    "examples",
    "example",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
    "externalDocs",
}


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


def _has_refs(schema_part: Any) -> bool:
    """Check if a schema part contains any $ref references."""
    if not isinstance(schema_part, dict):
        if isinstance(schema_part, list):
            return any(_has_refs(item) for item in schema_part)
        return False

    if "$ref" in schema_part:
        return True

    return any(_has_refs(v) for v in schema_part.values())


def _shallow_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Create a shallow copy of a dict, only copying the top level."""
    return {k: v for k, v in d.items()}


def _find_unresolved_refs(
    schema_part: Any,
    found_refs: Set[str],
    path: str = "",
) -> None:
    """
    Recursively find all unresolved $ref in a schema.

    Args:
        schema_part: The schema part to search
        found_refs: Set to collect found unresolved refs
        path: Current path in the schema (for debugging)
    """
    if not isinstance(schema_part, dict):
        if isinstance(schema_part, list):
            for i, item in enumerate(schema_part):
                _find_unresolved_refs(item, found_refs, f"{path}[{i}]")
        return

    if "$ref" in schema_part:
        # Check if this is an unresolved reference (not marked as circular)
        if not schema_part.get("_circular", False):
            found_refs.add(schema_part["$ref"])

    for key, value in schema_part.items():
        if key != "$ref" and key != "_circular":
            _find_unresolved_refs(value, found_refs, f"{path}.{key}" if path else key)


def validate_resolved_schema(
    schema: Dict[str, Any],
    raise_on_unresolved: bool = False,
) -> Set[str]:
    """
    Validate that a resolved schema has no remaining unresolved references.

    This function checks for any $ref that wasn't properly resolved.
    Circular references (marked with _circular=True) are allowed.

    Args:
        schema: The resolved schema to validate
        raise_on_unresolved: If True, raise UnresolvedReferenceError when unresolved refs found

    Returns:
        Set of unresolved reference paths found

    Raises:
        UnresolvedReferenceError: If raise_on_unresolved=True and unresolved refs are found
    """
    unresolved_refs: Set[str] = set()
    _find_unresolved_refs(schema, unresolved_refs)

    if unresolved_refs:
        logger.warning(
            f"Found {len(unresolved_refs)} unresolved reference(s) in resolved schema: "
            f"{', '.join(sorted(unresolved_refs))}"
        )
        if raise_on_unresolved:
            raise UnresolvedReferenceError(unresolved_refs)

    return unresolved_refs


def _merge_schema_with_metadata(
    resolved_schema: Dict[str, Any],
    original_schema: Dict[str, Any],
    context: ReferenceResolutionContext,
    reference_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge a resolved schema with metadata from the original reference.

    This preserves important metadata fields (title, description, examples, etc.)
    that may be defined at the reference level.

    Args:
        resolved_schema: The resolved schema content
        original_schema: The original schema containing the $ref and possibly metadata
        context: Resolution context
        reference_schema: Complete schema for resolving nested refs

    Returns:
        Merged schema with metadata preserved
    """
    result = _shallow_copy_dict(resolved_schema)

    # Preserve metadata from the original schema (the one with $ref)
    # These override the resolved schema's metadata if present
    for key, value in original_schema.items():
        if key == "$ref":
            continue
        if key in SCHEMA_METADATA_FIELDS:
            # Metadata at reference level takes precedence
            result[key] = value
        elif key not in result:
            # Other properties are added if not already present
            result[key] = _resolve_schema_references_internal(
                value, reference_schema, context
            )
        else:
            # Resolve the value if it's a complex type
            if isinstance(value, (dict, list)):
                result[key] = _resolve_schema_references_internal(
                    value, reference_schema, context
                )

    return result


def _resolve_schema_references_internal(
    schema_part: Any,
    reference_schema: Dict[str, Any],
    context: ReferenceResolutionContext,
) -> Any:
    """
    Internal recursive function to resolve schema references.

    Optimized for memory efficiency:
    - Uses shallow copies where possible
    - Only deep copies when necessary (schemas with nested mutable structures)
    - Caches resolved references to avoid redundant resolution
    - Tracks cache hits for performance monitoring

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
            # Only process list if it might contain refs
            if any(isinstance(item, (dict, list)) for item in schema_part):
                return [
                    _resolve_schema_references_internal(item, reference_schema, context)
                    for item in schema_part
                ]
            return schema_part
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

    # Handle $ref directly in the schema
    if "$ref" in schema_part:
        ref_path = schema_part["$ref"]
        context.reference_count += 1

        # Check for circular reference
        if ref_path in context.resolution_stack:
            context.circular_refs.add(ref_path)
            # Log circular reference with appropriate level based on first occurrence
            if len(context.circular_refs) == 1 or ref_path not in context.circular_refs:
                logger.warning(
                    f"Circular reference detected in OpenAPI schema: {ref_path}. "
                    "This may indicate a recursive data structure in your API models. "
                    "The reference will be preserved but marked as circular."
                )
            else:
                logger.debug(f"Circular reference encountered again: {ref_path}")
            # Return a placeholder for circular references
            return {"$ref": ref_path, "_circular": True}

        # Check cache first - optimized caching strategy
        if ref_path in context.resolved_cache:
            context.cache_hits += 1
            cached = context.resolved_cache[ref_path]

            # Only deep copy if the cached schema has nested mutable structures
            # that could be modified
            if ref_path in context.needs_deep_copy:
                result = copy.deepcopy(cached)
            else:
                # Shallow copy is sufficient for simple schemas
                result = _shallow_copy_dict(cached) if isinstance(cached, dict) else cached

            # Merge metadata and additional properties from the original schema
            if isinstance(result, dict) and len(schema_part) > 1:
                # There are additional properties besides $ref
                result = _merge_schema_with_metadata(
                    result, schema_part, context, reference_schema
                )

            return result

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

                # Determine if this schema needs deep copy when retrieved from cache
                # Only mark for deep copy if it has nested objects/arrays that could be mutated
                if isinstance(resolved_copy, dict):
                    has_mutable_nested = any(
                        isinstance(v, (dict, list))
                        for v in resolved_copy.values()
                    )
                    if has_mutable_nested:
                        context.needs_deep_copy.add(ref_path)

                # Cache the resolved schema (store original, not a copy)
                context.resolved_cache[ref_path] = resolved_copy

                # Create result with merged metadata and additional properties
                if isinstance(resolved_copy, dict):
                    # Merge metadata from original schema if there are additional properties
                    if len(schema_part) > 1:
                        result = _merge_schema_with_metadata(
                            resolved_copy, schema_part, context, reference_schema
                        )
                    else:
                        # No additional properties, just shallow copy
                        result = _shallow_copy_dict(resolved_copy)
                    return result
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
                logger.warning(
                    f"Unresolved reference in OpenAPI schema: {ref_path}. "
                    "This reference will be kept as-is but may cause issues during tool generation."
                )
            return _shallow_copy_dict(schema_part)

    # No $ref - process all keys recursively
    # Use shallow copy and only recurse into values that need processing
    context.depth += 1
    try:
        result = {}
        for key, value in schema_part.items():
            if isinstance(value, dict):
                result[key] = _resolve_schema_references_internal(
                    value, reference_schema, context
                )
            elif isinstance(value, list):
                # Only process list items that could contain refs
                if any(isinstance(item, (dict, list)) for item in value):
                    result[key] = [
                        _resolve_schema_references_internal(item, reference_schema, context)
                        if isinstance(item, (dict, list))
                        else item
                        for item in value
                    ]
                else:
                    result[key] = value
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

    # Log summary of resolution
    if context.circular_refs:
        circular_list = ", ".join(sorted(context.circular_refs))
        logger.warning(
            f"OpenAPI schema resolution completed with {len(context.circular_refs)} "
            f"circular reference(s): {circular_list}. "
            "These schemas reference themselves directly or indirectly. "
            "Consider reviewing your API models for unintended recursion."
        )

    if context.warnings:
        logger.info(
            f"Reference resolution completed with {len(context.warnings)} warning(s). "
            f"Resolved {context.reference_count} references with {context.cache_hits} cache hits."
        )
    elif context.reference_count > 0:
        logger.debug(
            f"Reference resolution completed successfully. "
            f"Resolved {context.reference_count} references with {context.cache_hits} cache hits "
            f"({context.cache_hits * 100 // max(context.reference_count, 1)}% cache hit rate)."
        )

    return result if isinstance(result, dict) else schema_part


def resolve_schema_references_with_details(
    schema_part: Dict[str, Any],
    reference_schema: Dict[str, Any],
    validate: bool = True,
) -> ReferenceResolutionResult:
    """
    Resolve schema references and return detailed information about the resolution.

    This is useful for debugging and for detecting problematic reference patterns.

    Args:
        schema_part: The part of the schema being processed that may contain references
        reference_schema: The complete schema used to resolve references from
        validate: If True, validate the resolved schema for unresolved references

    Returns:
        A ReferenceResolutionResult containing the resolved schema and metadata
    """
    context = ReferenceResolutionContext()
    result = _resolve_schema_references_internal(schema_part, reference_schema, context)
    resolved_schema = result if isinstance(result, dict) else schema_part

    # Validate the resolved schema for any remaining unresolved references
    unresolved_refs: Set[str] = set()
    if validate:
        unresolved_refs = validate_resolved_schema(resolved_schema, raise_on_unresolved=False)

    # Log circular references if any were found
    if context.circular_refs:
        circular_list = ", ".join(sorted(context.circular_refs))
        logger.warning(
            f"OpenAPI schema resolution completed with {len(context.circular_refs)} "
            f"circular reference(s): {circular_list}. "
            "These schemas reference themselves directly or indirectly."
        )

    return ReferenceResolutionResult(
        schema=resolved_schema,
        circular_refs=context.circular_refs,
        warnings=context.warnings,
        reference_count=context.reference_count,
        cache_hits=context.cache_hits,
        unresolved_refs=unresolved_refs,
    )


@dataclass
class SchemaAnalysisResult:
    """Result of schema analysis for problematic patterns."""

    # All unique $ref paths found in the schema
    all_refs: Set[str] = field(default_factory=set)
    # References that form circular dependencies
    circular_refs: Set[str] = field(default_factory=set)
    # References that could not be resolved
    unresolved_refs: Set[str] = field(default_factory=set)
    # External references (not starting with #/)
    external_refs: Set[str] = field(default_factory=set)
    # Maximum nesting depth of references
    max_depth: int = 0
    # Total number of reference usages
    total_ref_count: int = 0
    # Warnings about potential issues
    warnings: List[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        """Check if any problematic patterns were detected."""
        return bool(
            self.circular_refs
            or self.unresolved_refs
            or self.external_refs
            or self.warnings
        )


def _analyze_refs_recursive(
    schema_part: Any,
    reference_schema: Dict[str, Any],
    result: SchemaAnalysisResult,
    visited: Set[str],
    current_path: List[str],
    depth: int,
) -> None:
    """Recursively analyze schema for reference patterns."""
    if not isinstance(schema_part, dict):
        if isinstance(schema_part, list):
            for item in schema_part:
                _analyze_refs_recursive(
                    item, reference_schema, result, visited, current_path, depth
                )
        return

    result.max_depth = max(result.max_depth, depth)

    if "$ref" in schema_part:
        ref_path = schema_part["$ref"]
        result.total_ref_count += 1
        result.all_refs.add(ref_path)

        # Check for external references
        if not ref_path.startswith("#/"):
            result.external_refs.add(ref_path)
            if f"External reference not supported: {ref_path}" not in result.warnings:
                result.warnings.append(f"External reference not supported: {ref_path}")
            return

        # Check for circular reference
        if ref_path in current_path:
            result.circular_refs.add(ref_path)
            return

        # Check if reference can be resolved
        resolved = _resolve_json_pointer(reference_schema, ref_path)
        if resolved is None:
            result.unresolved_refs.add(ref_path)
            if f"Unresolved reference: {ref_path}" not in result.warnings:
                result.warnings.append(f"Unresolved reference: {ref_path}")
            return

        # Continue analyzing the resolved schema
        if ref_path not in visited:
            visited.add(ref_path)
            _analyze_refs_recursive(
                resolved,
                reference_schema,
                result,
                visited,
                current_path + [ref_path],
                depth + 1,
            )

    # Analyze nested structures
    for key, value in schema_part.items():
        if key != "$ref":
            _analyze_refs_recursive(
                value, reference_schema, result, visited, current_path, depth + 1
            )


def analyze_schema_references(
    schema: Dict[str, Any],
    reference_schema: Optional[Dict[str, Any]] = None,
) -> SchemaAnalysisResult:
    """
    Analyze a schema for problematic reference patterns.

    This function detects:
    - Circular references
    - Unresolved references
    - External references (not supported)
    - Deep nesting that could cause performance issues

    Args:
        schema: The schema to analyze
        reference_schema: The complete schema for resolving references.
                         If None, uses schema itself.

    Returns:
        SchemaAnalysisResult with details about detected patterns
    """
    if reference_schema is None:
        reference_schema = schema

    result = SchemaAnalysisResult()
    visited: Set[str] = set()

    _analyze_refs_recursive(schema, reference_schema, result, visited, [], 0)

    # Add warnings for deep nesting
    if result.max_depth > MAX_REFERENCE_DEPTH // 2:
        result.warnings.append(
            f"Deep reference nesting detected (depth: {result.max_depth}). "
            f"This may cause performance issues."
        )

    # Add warnings for high reference count
    if result.total_ref_count > MAX_REFERENCE_COUNT // 2:
        result.warnings.append(
            f"High reference count detected ({result.total_ref_count}). "
            f"This may cause performance issues."
        )

    return result


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
