import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Maximum depth for reference resolution to prevent stack overflow
MAX_REFERENCE_DEPTH = 50

# Maximum number of references to resolve before considering it problematic
MAX_REFERENCES_THRESHOLD = 1000


@dataclass
class ReferenceResolutionResult:
    """Result of reference resolution with diagnostic information."""

    schema: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    circular_refs_detected: Set[str] = field(default_factory=set)
    max_depth_reached: bool = False
    total_refs_resolved: int = 0


@dataclass
class _ResolutionContext:
    """Internal context for tracking resolution state."""

    reference_schema: Dict[str, Any]
    resolution_stack: Set[str] = field(default_factory=set)
    resolved_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    circular_refs_detected: Set[str] = field(default_factory=set)
    max_depth_reached: bool = False
    total_refs_resolved: int = 0
    current_depth: int = 0


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


def _get_ref_target(ref_path: str, reference_schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Resolve a JSON pointer reference to the target schema.

    Supports multiple reference formats:
    - #/components/schemas/ModelName
    - #/components/parameters/ParamName
    - #/components/requestBodies/BodyName
    - #/components/responses/ResponseName

    Args:
        ref_path: The $ref path (e.g., "#/components/schemas/User")
        reference_schema: The root schema to resolve from

    Returns:
        The resolved schema or None if not found
    """
    if not ref_path.startswith("#/"):
        return None

    # Parse the JSON pointer path
    parts = ref_path[2:].split("/")
    current = reference_schema

    for part in parts:
        # Handle URL-encoded characters in JSON pointers
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None

    return current if isinstance(current, dict) else None


def _resolve_schema_references_internal(
    schema_part: Dict[str, Any],
    context: _ResolutionContext,
) -> Dict[str, Any]:
    """
    Internal recursive function to resolve schema references.

    Args:
        schema_part: The part of the schema being processed
        context: Resolution context tracking state

    Returns:
        The schema with references resolved
    """
    # Check depth limit
    if context.current_depth > MAX_REFERENCE_DEPTH:
        if not context.max_depth_reached:
            context.max_depth_reached = True
            context.warnings.append(
                f"Maximum reference depth ({MAX_REFERENCE_DEPTH}) exceeded. "
                "Schema may have deeply nested or circular references."
            )
        return schema_part.copy()

    # Check if we've resolved too many references
    if context.total_refs_resolved > MAX_REFERENCES_THRESHOLD:
        if len(context.warnings) == 0 or "excessive references" not in context.warnings[-1]:
            context.warnings.append(
                f"Excessive number of references ({context.total_refs_resolved}+) detected. "
                "This may indicate problematic schema patterns."
            )
        return schema_part.copy()

    # Make a copy to avoid modifying the input schema
    schema_part = schema_part.copy()

    # Handle $ref directly in the schema
    if "$ref" in schema_part:
        ref_path = schema_part["$ref"]

        # Check for circular reference
        if ref_path in context.resolution_stack:
            context.circular_refs_detected.add(ref_path)
            logger.debug(f"Circular reference detected: {ref_path}")
            # Return a placeholder that indicates circular reference
            # Keep the $ref but add a marker
            return {
                "$ref": ref_path,
                "x-circular-ref": True,
                "description": schema_part.get("description", f"Circular reference to {ref_path}"),
            }

        # Check if already resolved and cached
        if ref_path in context.resolved_cache:
            # Return a deep copy of cached result to prevent mutations
            import copy

            resolved = copy.deepcopy(context.resolved_cache[ref_path])
            schema_part.pop("$ref")
            schema_part.update(resolved)
            return schema_part

        # Try to resolve the reference
        ref_target = _get_ref_target(ref_path, context.reference_schema)

        if ref_target is not None:
            context.total_refs_resolved += 1

            # Add to resolution stack to detect cycles
            context.resolution_stack.add(ref_path)
            context.current_depth += 1

            try:
                # Recursively resolve the target schema
                resolved_target = _resolve_schema_references_internal(
                    ref_target,
                    context,
                )

                # Cache the resolved schema
                context.resolved_cache[ref_path] = resolved_target

                # Remove the $ref and merge with resolved schema
                schema_part.pop("$ref")
                schema_part.update(resolved_target)
            finally:
                # Always remove from stack when done
                context.resolution_stack.discard(ref_path)
                context.current_depth -= 1
        else:
            # Reference not found - log warning but keep the $ref
            context.warnings.append(f"Unresolved reference: {ref_path}")
            logger.warning(f"Could not resolve reference: {ref_path}")

    # Recursively resolve references in all dictionary values
    for key, value in list(schema_part.items()):
        if isinstance(value, dict):
            context.current_depth += 1
            try:
                schema_part[key] = _resolve_schema_references_internal(value, context)
            finally:
                context.current_depth -= 1
        elif isinstance(value, list):
            # Only process list items that are dictionaries since only they can contain refs
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    context.current_depth += 1
                    try:
                        new_list.append(_resolve_schema_references_internal(item, context))
                    finally:
                        context.current_depth -= 1
                else:
                    new_list.append(item)
            schema_part[key] = new_list

    return schema_part


def resolve_schema_references(schema_part: Dict[str, Any], reference_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve schema references in OpenAPI schemas.

    This function handles:
    - Circular references (detected and marked)
    - Deeply nested references (with depth limit protection)
    - Caching of resolved schemas for efficiency
    - Multiple reference types (schemas, parameters, requestBodies, responses)

    Args:
        schema_part: The part of the schema being processed that may contain references
        reference_schema: The complete schema used to resolve references from

    Returns:
        The schema with references resolved
    """
    context = _ResolutionContext(reference_schema=reference_schema)
    result = _resolve_schema_references_internal(schema_part, context)

    # Log any warnings
    for warning in context.warnings:
        logger.warning(warning)

    if context.circular_refs_detected:
        logger.info(f"Circular references detected: {context.circular_refs_detected}")

    return result


def resolve_schema_references_with_diagnostics(
    schema_part: Dict[str, Any],
    reference_schema: Dict[str, Any],
) -> ReferenceResolutionResult:
    """
    Resolve schema references with full diagnostic information.

    Use this function when you need detailed information about the resolution process,
    including warnings, circular reference detection, and statistics.

    Args:
        schema_part: The part of the schema being processed that may contain references
        reference_schema: The complete schema used to resolve references from

    Returns:
        ReferenceResolutionResult containing the resolved schema and diagnostic info
    """
    context = _ResolutionContext(reference_schema=reference_schema)
    resolved_schema = _resolve_schema_references_internal(schema_part, context)

    return ReferenceResolutionResult(
        schema=resolved_schema,
        warnings=context.warnings,
        circular_refs_detected=context.circular_refs_detected,
        max_depth_reached=context.max_depth_reached,
        total_refs_resolved=context.total_refs_resolved,
    )


def detect_problematic_references(openapi_schema: Dict[str, Any]) -> List[str]:
    """
    Analyze an OpenAPI schema for problematic reference patterns.

    This function performs a dry-run analysis to detect potential issues
    without fully resolving the schema.

    Detects:
    - Direct circular references (A -> A)
    - Indirect circular references (A -> B -> A)
    - Deep reference chains (A -> B -> C -> D -> ...)
    - Missing reference targets
    - Excessive reference count

    Args:
        openapi_schema: The complete OpenAPI schema to analyze

    Returns:
        List of warning messages describing problematic patterns
    """
    warnings: List[str] = []
    schemas = openapi_schema.get("components", {}).get("schemas", {})

    if not schemas:
        return warnings

    # Build reference graph
    ref_graph: Dict[str, Set[str]] = {}
    for schema_name, schema_def in schemas.items():
        refs = _collect_refs_from_schema(schema_def)
        ref_graph[schema_name] = refs

    # Detect circular references using DFS
    circular_refs = _find_circular_refs(ref_graph)
    for cycle in circular_refs:
        warnings.append(f"Circular reference chain detected: {' -> '.join(cycle)}")

    # Detect deep reference chains
    max_chain_length = 0
    deepest_chain: List[str] = []
    for schema_name in schemas:
        chain = _find_longest_chain(schema_name, ref_graph, set())
        if len(chain) > max_chain_length:
            max_chain_length = len(chain)
            deepest_chain = chain

    if max_chain_length > 10:
        warnings.append(
            f"Deep reference chain detected (length {max_chain_length}): "
            f"{' -> '.join(deepest_chain[:5])}{'...' if len(deepest_chain) > 5 else ''}"
        )

    # Check for missing reference targets
    all_refs: Set[str] = set()
    for refs in ref_graph.values():
        all_refs.update(refs)

    for ref in all_refs:
        if ref not in schemas:
            warnings.append(f"Reference to undefined schema: {ref}")

    # Check total reference count
    total_refs = sum(len(refs) for refs in ref_graph.values())
    if total_refs > 500:
        warnings.append(
            f"High reference count ({total_refs}) may impact performance. "
            "Consider simplifying the schema structure."
        )

    return warnings


def _collect_refs_from_schema(schema: Dict[str, Any], visited: Optional[Set[int]] = None) -> Set[str]:
    """Collect all schema references from a schema definition."""
    if visited is None:
        visited = set()

    # Prevent infinite recursion on the same dict object
    schema_id = id(schema)
    if schema_id in visited:
        return set()
    visited.add(schema_id)

    refs: Set[str] = set()

    if "$ref" in schema:
        ref_path = schema["$ref"]
        if ref_path.startswith("#/components/schemas/"):
            refs.add(ref_path.split("/")[-1])

    for key, value in schema.items():
        if isinstance(value, dict):
            refs.update(_collect_refs_from_schema(value, visited))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    refs.update(_collect_refs_from_schema(item, visited))

    return refs


def _find_circular_refs(ref_graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Find all circular reference chains using DFS."""
    cycles: List[List[str]] = []
    visited: Set[str] = set()
    rec_stack: List[str] = []

    def dfs(node: str) -> None:
        if node in rec_stack:
            # Found a cycle
            cycle_start = rec_stack.index(node)
            cycle = rec_stack[cycle_start:] + [node]
            cycles.append(cycle)
            return

        if node in visited:
            return

        visited.add(node)
        rec_stack.append(node)

        for neighbor in ref_graph.get(node, set()):
            dfs(neighbor)

        rec_stack.pop()

    for node in ref_graph:
        if node not in visited:
            dfs(node)

    return cycles


def _find_longest_chain(
    node: str,
    ref_graph: Dict[str, Set[str]],
    visited: Set[str],
) -> List[str]:
    """Find the longest reference chain starting from a node."""
    if node in visited or node not in ref_graph:
        return []

    visited.add(node)
    longest: List[str] = [node]

    for neighbor in ref_graph.get(node, set()):
        chain = [node] + _find_longest_chain(neighbor, ref_graph, visited.copy())
        if len(chain) > len(longest):
            longest = chain

    return longest


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
