from typing import Any, Dict, List, Tuple


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


def resolve_schema_references(schema_part: Dict[str, Any], reference_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve schema references in OpenAPI schemas.

    Args:
        schema_part: The part of the schema being processed that may contain references
        reference_schema: The complete schema used to resolve references from

    Returns:
        The schema with references resolved
    """
    # Make a copy to avoid modifying the input schema
    schema_part = schema_part.copy()

    # Handle $ref directly in the schema
    if "$ref" in schema_part:
        ref_path = schema_part["$ref"]
        # Standard OpenAPI references are in the format "#/components/schemas/ModelName"
        if ref_path.startswith("#/components/schemas/"):
            model_name = ref_path.split("/")[-1]
            if "components" in reference_schema and "schemas" in reference_schema["components"]:
                if model_name in reference_schema["components"]["schemas"]:
                    # Replace with the resolved schema
                    ref_schema = reference_schema["components"]["schemas"][model_name].copy()
                    # Remove the $ref key and merge with the original schema
                    schema_part.pop("$ref")
                    schema_part.update(ref_schema)

    # Recursively resolve references in all dictionary values
    for key, value in schema_part.items():
        if isinstance(value, dict):
            schema_part[key] = resolve_schema_references(value, reference_schema)
        elif isinstance(value, list):
            # Only process list items that are dictionaries since only they can contain refs
            schema_part[key] = [
                resolve_schema_references(item, reference_schema) if isinstance(item, dict) else item for item in value
            ]

    return schema_part


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


def generate_example_from_schema(schema: Dict[str, Any], prop_name: str = "") -> Any:
    """
    Generate a realistic example response from a JSON schema.

    Args:
        schema: The JSON schema to generate an example from
        prop_name: The property name (used to generate contextual examples)

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
            for name, prop_schema in schema["properties"].items():
                # Pass property name to generate contextual examples
                prop_example = generate_example_from_schema(prop_schema, name)
                if prop_example is not None:
                    result[name] = prop_example
        return result

    elif schema_type == "array":
        if "items" in schema:
            # Generate a single example item
            item_example = generate_example_from_schema(schema["items"], prop_name)
            if item_example is not None:
                return [item_example]
        return []

    elif schema_type == "string":
        # Check if there's a format
        format_type = schema.get("format")
        if format_type == "date-time":
            return "2024-01-15T09:30:00Z"
        elif format_type == "date":
            return "2024-01-15"
        elif format_type == "time":
            return "09:30:00"
        elif format_type == "email":
            return "user@example.com"
        elif format_type == "uri" or format_type == "url":
            return "https://api.example.com/resource"
        elif format_type == "uuid":
            return "550e8400-e29b-41d4-a716-446655440000"
        elif format_type == "hostname":
            return "api.example.com"
        elif format_type == "ipv4":
            return "192.168.1.1"
        elif format_type == "ipv6":
            return "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

        # Check for enum values
        if "enum" in schema and schema["enum"]:
            return schema["enum"][0]

        # Generate contextual examples based on property name
        name_lower = prop_name.lower() if prop_name else ""
        if "id" in name_lower:
            return "abc123"
        elif "name" in name_lower:
            return "Example Name"
        elif "title" in name_lower:
            return "Example Title"
        elif "description" in name_lower or "desc" in name_lower:
            return "A detailed description of the item."
        elif "email" in name_lower:
            return "user@example.com"
        elif "phone" in name_lower or "tel" in name_lower:
            return "+1-555-123-4567"
        elif "url" in name_lower or "link" in name_lower or "href" in name_lower:
            return "https://example.com"
        elif "address" in name_lower:
            return "123 Main Street"
        elif "city" in name_lower:
            return "San Francisco"
        elif "country" in name_lower:
            return "United States"
        elif "zip" in name_lower or "postal" in name_lower:
            return "94102"
        elif "status" in name_lower:
            return "active"
        elif "type" in name_lower or "kind" in name_lower:
            return "standard"
        elif "tag" in name_lower:
            return "example-tag"
        elif "key" in name_lower:
            return "example_key"
        elif "token" in name_lower:
            return "eyJhbGciOiJIUzI1NiIs..."
        elif "password" in name_lower or "secret" in name_lower:
            return "********"
        elif "message" in name_lower or "text" in name_lower or "content" in name_lower:
            return "This is an example message."
        elif "path" in name_lower or "file" in name_lower:
            return "/path/to/file.txt"
        elif "color" in name_lower or "colour" in name_lower:
            return "#3498db"
        elif "version" in name_lower:
            return "1.0.0"
        elif "code" in name_lower:
            return "ABC123"
        elif "currency" in name_lower:
            return "USD"
        elif "lang" in name_lower or "locale" in name_lower:
            return "en-US"

        # Use title if available, otherwise return a generic but meaningful value
        title = schema.get("title", "")
        if title and title.lower() != "string":
            return f"example_{title.lower().replace(' ', '_')}"

        return "example_value"

    elif schema_type == "integer":
        # Generate contextual integer examples
        name_lower = prop_name.lower() if prop_name else ""
        if "id" in name_lower:
            return 12345
        elif "count" in name_lower or "total" in name_lower or "num" in name_lower:
            return 42
        elif "age" in name_lower:
            return 30
        elif "year" in name_lower:
            return 2024
        elif "month" in name_lower:
            return 6
        elif "day" in name_lower:
            return 15
        elif "port" in name_lower:
            return 8080
        elif "page" in name_lower:
            return 1
        elif "limit" in name_lower or "size" in name_lower:
            return 10
        elif "offset" in name_lower or "skip" in name_lower:
            return 0
        elif "quantity" in name_lower or "qty" in name_lower:
            return 5
        elif "priority" in name_lower or "order" in name_lower or "index" in name_lower:
            return 1
        return 42

    elif schema_type == "number":
        # Generate contextual number examples
        name_lower = prop_name.lower() if prop_name else ""
        if "price" in name_lower or "cost" in name_lower or "amount" in name_lower:
            return 99.99
        elif "rate" in name_lower or "percent" in name_lower:
            return 0.15
        elif "lat" in name_lower:
            return 37.7749
        elif "lon" in name_lower or "lng" in name_lower:
            return -122.4194
        elif "weight" in name_lower:
            return 2.5
        elif "height" in name_lower or "width" in name_lower or "length" in name_lower:
            return 10.0
        elif "score" in name_lower or "rating" in name_lower:
            return 4.5
        elif "temperature" in name_lower or "temp" in name_lower:
            return 72.5
        return 123.45

    elif schema_type == "boolean":
        # Generate contextual boolean examples
        name_lower = prop_name.lower() if prop_name else ""
        if "enabled" in name_lower or "active" in name_lower or "is_" in name_lower:
            return True
        elif "disabled" in name_lower or "deleted" in name_lower or "archived" in name_lower:
            return False
        return True

    elif schema_type == "null":
        return None

    # Default case
    return None


def generate_param_description(param_name: str, param_type: str, param_source: str = "") -> str:
    """
    Generate a sensible default description for a parameter based on its name and type.

    Args:
        param_name: The parameter name
        param_type: The parameter type (string, integer, etc.)
        param_source: The parameter source (path, query, body)

    Returns:
        A generated description string, or empty string if no match
    """
    name_lower = param_name.lower()

    # Common ID patterns
    if name_lower == "id" or name_lower.endswith("_id"):
        entity = name_lower.replace("_id", "").replace("id", "item")
        if entity:
            return f"Unique identifier for the {entity.replace('_', ' ')}"
        return "Unique identifier"

    # Pagination parameters
    if name_lower in ("page", "page_number", "pagenumber"):
        return "Page number for pagination (starts at 1)"
    if name_lower in ("limit", "page_size", "pagesize", "per_page", "perpage"):
        return "Maximum number of items to return"
    if name_lower in ("offset", "skip"):
        return "Number of items to skip"

    # Search and filtering
    if name_lower in ("q", "query", "search", "search_query", "searchquery"):
        return "Search query string"
    if name_lower in ("filter", "filters"):
        return "Filter criteria"
    if name_lower in ("sort", "sort_by", "sortby", "order_by", "orderby"):
        return "Field to sort results by"
    if name_lower in ("order", "sort_order", "sortorder", "direction", "sort_direction"):
        return "Sort direction (asc or desc)"

    # Common field names
    if name_lower == "name":
        return "Name or title"
    if name_lower == "email":
        return "Email address"
    if name_lower == "username":
        return "Username for the account"
    if name_lower == "password":
        return "Password for authentication"
    if name_lower == "description":
        return "Detailed description"
    if name_lower == "title":
        return "Title or heading"
    if name_lower in ("content", "body", "text", "message"):
        return "Content or message text"
    if name_lower == "url":
        return "URL or web address"
    if name_lower == "status":
        return "Current status"
    if name_lower == "type":
        return "Type or category"
    if name_lower in ("tag", "tags"):
        return "Tags for categorization"
    if name_lower in ("category", "categories"):
        return "Category or classification"
    if name_lower == "price":
        return "Price amount"
    if name_lower == "quantity":
        return "Quantity or count"
    if name_lower in ("date", "created_at", "updated_at", "timestamp"):
        return "Date/time value"
    if name_lower in ("start_date", "startdate", "from_date", "fromdate"):
        return "Start date for filtering"
    if name_lower in ("end_date", "enddate", "to_date", "todate"):
        return "End date for filtering"
    if name_lower in ("active", "enabled", "is_active", "is_enabled"):
        return "Whether the item is active/enabled"
    if name_lower in ("include_details", "includedetails", "detailed", "verbose"):
        return "Include additional details in response"
    if name_lower == "fields":
        return "Fields to include in response"
    if name_lower in ("format", "output_format"):
        return "Output format"

    # Address fields
    if name_lower == "address":
        return "Street address"
    if name_lower == "city":
        return "City name"
    if name_lower in ("state", "province", "region"):
        return "State or province"
    if name_lower == "country":
        return "Country name or code"
    if name_lower in ("zip", "zipcode", "zip_code", "postal_code", "postalcode"):
        return "Postal/ZIP code"

    # Contact info
    if name_lower in ("phone", "phone_number", "phonenumber", "tel", "telephone"):
        return "Phone number"

    # File-related
    if name_lower in ("file", "filename", "file_name"):
        return "File name"
    if name_lower in ("path", "filepath", "file_path"):
        return "File path"

    # For path parameters, add context
    if param_source == "path":
        # Convert snake_case to readable text
        readable_name = param_name.replace("_", " ")
        return f"The {readable_name} to operate on"

    return ""


def format_parameter_docs(
    params: List[Tuple[str, Dict[str, Any]]],
    required_params: List[str],
    section_title: str,
) -> str:
    """
    Format parameter documentation for tool descriptions.

    Args:
        params: List of tuples containing (param_name, param_info)
        required_params: List of required parameter names
        section_title: Title for this parameter section (e.g., "Path Parameters")

    Returns:
        Formatted markdown string for the parameters section
    """
    if not params:
        return ""

    # Separate required and optional parameters
    required = []
    optional = []

    for param_name, param_info in params:
        param_schema = param_info.get("schema", param_info)
        param_type = get_single_param_type_from_schema(param_schema)
        param_desc = param_info.get("description", "") or param_schema.get("description", "")
        default = param_schema.get("default")

        param_entry = {
            "name": param_name,
            "type": param_type,
            "description": param_desc,
            "default": default,
            "has_default": "default" in param_schema,
        }

        if param_name in required_params:
            required.append(param_entry)
        else:
            optional.append(param_entry)

    result = f"\n\n### {section_title}:\n"

    # Format required parameters first
    if required:
        result += "\n**Required:**\n"
        for param in required:
            result += _format_single_param(param, is_required=True)

    # Format optional parameters
    if optional:
        result += "\n**Optional:**\n"
        for param in optional:
            result += _format_single_param(param, is_required=False)

    return result


def _format_single_param(param: Dict[str, Any], is_required: bool = False) -> str:
    """
    Format a single parameter entry for display.

    Args:
        param: Dictionary with parameter details (name, type, description, default, has_default)
        is_required: Whether this parameter is required

    Returns:
        Formatted string for the parameter
    """
    # Start with name and type
    line = f"- `{param['name']}` ({param['type']})"

    # Add required/optional indicator
    if is_required:
        line += " *required*"

    # Add default value if present
    if param["has_default"]:
        default_val = param["default"]
        if isinstance(default_val, str):
            line += f" [default: \"{default_val}\"]"
        elif default_val is None:
            line += " [default: null]"
        else:
            line += f" [default: {default_val}]"

    # Add description
    if param["description"]:
        line += f": {param['description']}"

    return line + "\n"


def format_all_parameters_docs(
    path_params: List[Tuple[str, Dict[str, Any]]],
    query_params: List[Tuple[str, Dict[str, Any]]],
    body_params: List[Tuple[str, Dict[str, Any]]],
    required_props: List[str],
) -> str:
    """
    Format all parameters into a documentation section, organized by parameter type.

    Parameters are grouped by their source (path, query, body) with clear headers,
    and within each group, required parameters are listed before optional ones.

    Args:
        path_params: List of path parameters
        query_params: List of query parameters
        body_params: List of body parameters
        required_props: List of all required parameter names

    Returns:
        Formatted markdown string for all parameters
    """
    if not path_params and not query_params and not body_params:
        return ""

    result = "\n\n### Parameters:\n"

    # Format path parameters
    if path_params:
        result += _format_param_group(path_params, required_props, "Path Parameters")

    # Format query parameters
    if query_params:
        result += _format_param_group(query_params, required_props, "Query Parameters")

    # Format body parameters
    if body_params:
        result += _format_param_group(body_params, required_props, "Body Parameters")

    return result


def _format_param_group(
    params: List[Tuple[str, Dict[str, Any]]],
    required_props: List[str],
    group_title: str,
) -> str:
    """
    Format a group of parameters with a title, separating required and optional.

    Args:
        params: List of tuples containing (param_name, param_info)
        required_props: List of all required parameter names
        group_title: Title for this parameter group

    Returns:
        Formatted markdown string for this parameter group
    """
    if not params:
        return ""

    # Determine parameter source from group title
    param_source = ""
    if "Path" in group_title:
        param_source = "path"
    elif "Query" in group_title:
        param_source = "query"
    elif "Body" in group_title:
        param_source = "body"

    # Separate required and optional parameters
    required = []
    optional = []

    for param_name, param_info in params:
        param_schema = param_info.get("schema", param_info)
        param_type = get_single_param_type_from_schema(param_schema)
        param_desc = param_info.get("description", "") or param_schema.get("description", "")

        # Generate default description if none provided
        if not param_desc:
            param_desc = generate_param_description(param_name, param_type, param_source)

        default = param_schema.get("default")

        param_entry = {
            "name": param_name,
            "type": param_type,
            "description": param_desc,
            "default": default,
            "has_default": "default" in param_schema,
        }

        if param_name in required_props:
            required.append(param_entry)
        else:
            optional.append(param_entry)

    result = f"\n**{group_title}:**\n"

    # Format required parameters first (always before optional)
    if required:
        for param in required:
            result += _format_single_param(param, is_required=True)

    # Format optional parameters (always after required)
    if optional:
        for param in optional:
            result += _format_single_param(param, is_required=False)

    return result


def generate_tool_summary(
    summary: str,
    description: str,
    method: str,
    path: str,
    required_count: int,
    optional_count: int,
) -> str:
    """
    Generate a concise summary section for a tool description.

    The summary provides a quick overview at the top of the description,
    making it easy to understand what the tool does at a glance.

    Args:
        summary: The OpenAPI operation summary
        description: The OpenAPI operation description
        method: HTTP method (GET, POST, etc.)
        path: The API endpoint path
        required_count: Number of required parameters
        optional_count: Number of optional parameters

    Returns:
        Formatted summary section string
    """
    # Build the main summary line
    if summary:
        main_summary = summary
    else:
        main_summary = f"{method.upper()} {path}"

    # Build a brief parameter hint
    param_hints = []
    if required_count > 0:
        param_hints.append(f"{required_count} required")
    if optional_count > 0:
        param_hints.append(f"{optional_count} optional")

    param_info = ""
    if param_hints:
        param_info = f" ({', '.join(param_hints)} params)"

    # Create the summary section
    result = f"**{main_summary}**{param_info}"

    # Add description as a secondary line if different from summary
    if description and description != summary:
        # Truncate long descriptions for the summary section
        if len(description) > 150:
            truncated = description[:147].rsplit(" ", 1)[0] + "..."
            result += f"\n\n{truncated}"
        else:
            result += f"\n\n{description}"

    return result
