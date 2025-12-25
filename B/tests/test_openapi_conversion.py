from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import mcp.types as types

from fastapi_mcp.openapi.convert import convert_openapi_to_mcp_tools
from fastapi_mcp.openapi.utils import (
    clean_schema_for_display,
    format_all_parameters_docs,
    format_parameter_docs,
    generate_default_param_description,
    generate_example_from_schema,
    generate_tool_summary,
    get_single_param_type_from_schema,
)


def test_simple_app_conversion(simple_fastapi_app: FastAPI):
    openapi_schema = get_openapi(
        title=simple_fastapi_app.title,
        version=simple_fastapi_app.version,
        openapi_version=simple_fastapi_app.openapi_version,
        description=simple_fastapi_app.description,
        routes=simple_fastapi_app.routes,
    )

    tools, operation_map = convert_openapi_to_mcp_tools(openapi_schema)

    assert len(tools) == 6
    assert len(operation_map) == 6

    expected_operations = ["list_items", "get_item", "create_item", "update_item", "delete_item", "raise_error"]
    for op in expected_operations:
        assert op in operation_map

    for tool in tools:
        assert isinstance(tool, types.Tool)
        assert tool.name in expected_operations
        assert tool.description is not None
        assert tool.inputSchema is not None


def test_complex_app_conversion(complex_fastapi_app: FastAPI):
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    tools, operation_map = convert_openapi_to_mcp_tools(openapi_schema)

    expected_operations = ["list_products", "get_product", "create_order", "get_customer"]
    assert len(tools) == len(expected_operations)
    assert len(operation_map) == len(expected_operations)

    for op in expected_operations:
        assert op in operation_map

    for tool in tools:
        assert isinstance(tool, types.Tool)
        assert tool.name in expected_operations
        assert tool.description is not None
        assert tool.inputSchema is not None


def test_describe_full_response_schema(simple_fastapi_app: FastAPI):
    openapi_schema = get_openapi(
        title=simple_fastapi_app.title,
        version=simple_fastapi_app.version,
        openapi_version=simple_fastapi_app.openapi_version,
        description=simple_fastapi_app.description,
        routes=simple_fastapi_app.routes,
    )

    tools_full, _ = convert_openapi_to_mcp_tools(openapi_schema, describe_full_response_schema=True)

    tools_simple, _ = convert_openapi_to_mcp_tools(openapi_schema, describe_full_response_schema=False)

    for i, tool in enumerate(tools_full):
        assert tool.description is not None
        assert tools_simple[i].description is not None

        tool_desc = tool.description or ""
        simple_desc = tools_simple[i].description or ""

        assert len(tool_desc) >= len(simple_desc)

        if tool.name == "delete_item":
            continue

        assert "**Output Schema:**" in tool_desc

        if "**Output Schema:**" in simple_desc:
            assert len(tool_desc) > len(simple_desc)


def test_describe_all_responses(complex_fastapi_app: FastAPI):
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    tools_all, _ = convert_openapi_to_mcp_tools(openapi_schema, describe_all_responses=True)

    tools_success, _ = convert_openapi_to_mcp_tools(openapi_schema, describe_all_responses=False)

    create_order_all = next(t for t in tools_all if t.name == "create_order")
    create_order_success = next(t for t in tools_success if t.name == "create_order")

    assert create_order_all.description is not None
    assert create_order_success.description is not None

    all_desc = create_order_all.description or ""
    success_desc = create_order_success.description or ""

    assert "400" in all_desc
    assert "404" in all_desc
    assert "422" in all_desc

    assert all_desc.count("400") >= success_desc.count("400")


def test_schema_utils():
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "name"],
        "additionalProperties": False,
        "x-internal": "Some internal data",
    }

    cleaned = clean_schema_for_display(schema)

    assert "required" in cleaned
    assert "properties" in cleaned
    assert "type" in cleaned

    example = generate_example_from_schema(schema)
    assert "id" in example
    assert "name" in example
    assert "tags" in example
    assert isinstance(example["id"], int)
    assert isinstance(example["name"], str)
    assert isinstance(example["tags"], list)

    assert get_single_param_type_from_schema({"type": "string"}) == "string"
    assert get_single_param_type_from_schema({"type": "array", "items": {"type": "string"}}) == "array"

    array_schema = {"type": "array", "items": {"type": "string", "enum": ["red", "green", "blue"]}}
    array_example = generate_example_from_schema(array_schema)
    assert isinstance(array_example, list)
    assert len(array_example) > 0

    assert isinstance(array_example[0], str)


def test_parameter_handling(complex_fastapi_app: FastAPI):
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    tools, operation_map = convert_openapi_to_mcp_tools(openapi_schema)

    list_products_tool = next(tool for tool in tools if tool.name == "list_products")

    properties = list_products_tool.inputSchema["properties"]

    assert "product_id" not in properties  # This is from get_product, not list_products

    assert "category" in properties
    assert properties["category"].get("type") == "string"  # Enum converted to string
    assert "description" in properties["category"]
    assert "Filter by product category" in properties["category"]["description"]

    assert "min_price" in properties
    assert properties["min_price"].get("type") == "number"
    assert "description" in properties["min_price"]
    assert "Minimum price filter" in properties["min_price"]["description"]
    if "minimum" in properties["min_price"]:
        assert properties["min_price"]["minimum"] > 0  # gt=0 in Query param

    assert "in_stock_only" in properties
    assert properties["in_stock_only"].get("type") == "boolean"
    assert properties["in_stock_only"].get("default") is False  # Default value preserved

    assert "page" in properties
    assert properties["page"].get("type") == "integer"
    assert properties["page"].get("default") == 1  # Default value preserved
    if "minimum" in properties["page"]:
        assert properties["page"]["minimum"] >= 1  # ge=1 in Query param

    assert "size" in properties
    assert properties["size"].get("type") == "integer"
    if "minimum" in properties["size"] and "maximum" in properties["size"]:
        assert properties["size"]["minimum"] >= 1  # ge=1 in Query param
        assert properties["size"]["maximum"] <= 100  # le=100 in Query param

    assert "tag" in properties
    assert properties["tag"].get("type") == "array"

    required = list_products_tool.inputSchema.get("required", [])
    assert "page" not in required  # Has default value
    assert "category" not in required  # Optional parameter

    assert "list_products" in operation_map
    assert operation_map["list_products"]["path"] == "/products"
    assert operation_map["list_products"]["method"] == "get"

    get_product_tool = next(tool for tool in tools if tool.name == "get_product")
    get_product_props = get_product_tool.inputSchema["properties"]

    assert "product_id" in get_product_props
    assert get_product_props["product_id"].get("type") == "string"  # UUID converted to string
    assert "description" in get_product_props["product_id"]

    get_customer_tool = next(tool for tool in tools if tool.name == "get_customer")
    get_customer_props = get_customer_tool.inputSchema["properties"]

    assert "fields" in get_customer_props
    assert get_customer_props["fields"].get("type") == "array"
    if "items" in get_customer_props["fields"]:
        assert get_customer_props["fields"]["items"].get("type") == "string"


def test_request_body_handling(complex_fastapi_app: FastAPI):
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    create_order_route = openapi_schema["paths"]["/orders"]["post"]
    original_request_body = create_order_route["requestBody"]["content"]["application/json"]["schema"]
    original_properties = original_request_body.get("properties", {})

    tools, operation_map = convert_openapi_to_mcp_tools(openapi_schema)

    create_order_tool = next(tool for tool in tools if tool.name == "create_order")

    properties = create_order_tool.inputSchema["properties"]

    assert "customer_id" in properties
    assert "items" in properties
    assert "shipping_address_id" in properties
    assert "payment_method" in properties
    assert "notes" in properties

    for param_name in ["customer_id", "items", "shipping_address_id", "payment_method", "notes"]:
        if "description" in original_properties.get(param_name, {}):
            assert "description" in properties[param_name]
            assert properties[param_name]["description"] == original_properties[param_name]["description"]

    for param_name in ["customer_id", "items", "shipping_address_id", "payment_method", "notes"]:
        assert properties[param_name]["title"] == param_name

    for param_name in ["customer_id", "items", "shipping_address_id", "payment_method", "notes"]:
        if "default" in original_properties.get(param_name, {}):
            assert "default" in properties[param_name]
            assert properties[param_name]["default"] == original_properties[param_name]["default"]

    required = create_order_tool.inputSchema.get("required", [])
    assert "customer_id" in required
    assert "items" in required
    assert "shipping_address_id" in required
    assert "payment_method" in required
    assert "notes" not in required  # Optional in OrderRequest

    assert properties["items"].get("type") == "array"
    if "items" in properties["items"]:
        item_props = properties["items"]["items"]
        assert item_props.get("type") == "object"
        if "properties" in item_props:
            assert "product_id" in item_props["properties"]
            assert "quantity" in item_props["properties"]
            assert "unit_price" in item_props["properties"]
            assert "total" in item_props["properties"]

            for nested_param in ["product_id", "quantity", "unit_price", "total"]:
                assert "title" in item_props["properties"][nested_param]

                # Check if the original nested schema had descriptions
                original_item_schema = original_properties.get("items", {}).get("items", {}).get("properties", {})
                if "description" in original_item_schema.get(nested_param, {}):
                    assert "description" in item_props["properties"][nested_param]
                    assert (
                        item_props["properties"][nested_param]["description"]
                        == original_item_schema[nested_param]["description"]
                    )

    assert "create_order" in operation_map
    assert operation_map["create_order"]["path"] == "/orders"
    assert operation_map["create_order"]["method"] == "post"


def test_missing_type_handling(complex_fastapi_app: FastAPI):
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    # Remove the type field from the product_id schema
    params = openapi_schema["paths"]["/products/{product_id}"]["get"]["parameters"]
    for param in params:
        if param.get("name") == "product_id" and "schema" in param:
            param["schema"].pop("type", None)
            break

    tools, operation_map = convert_openapi_to_mcp_tools(openapi_schema)

    get_product_tool = next(tool for tool in tools if tool.name == "get_product")
    get_product_props = get_product_tool.inputSchema["properties"]

    assert "product_id" in get_product_props
    assert get_product_props["product_id"].get("type") == "string"  # Default type applied


def test_body_params_descriptions_and_defaults(complex_fastapi_app: FastAPI):
    """
    Test that descriptions and defaults from request body parameters
    are properly transferred to the MCP tool schema properties.
    """
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    order_request_schema = openapi_schema["components"]["schemas"]["OrderRequest"]

    order_request_schema["properties"]["customer_id"]["description"] = "Test customer ID description"
    order_request_schema["properties"]["payment_method"]["description"] = "Test payment method description"
    order_request_schema["properties"]["notes"]["default"] = "Default order notes"

    item_schema = openapi_schema["components"]["schemas"]["OrderItem"]
    item_schema["properties"]["product_id"]["description"] = "Test product ID description"
    item_schema["properties"]["quantity"]["default"] = 1

    tools, _ = convert_openapi_to_mcp_tools(openapi_schema)

    create_order_tool = next(tool for tool in tools if tool.name == "create_order")
    properties = create_order_tool.inputSchema["properties"]

    assert "description" in properties["customer_id"]
    assert properties["customer_id"]["description"] == "Test customer ID description"

    assert "description" in properties["payment_method"]
    assert properties["payment_method"]["description"] == "Test payment method description"

    assert "default" in properties["notes"]
    assert properties["notes"]["default"] == "Default order notes"

    if "items" in properties:
        assert properties["items"]["type"] == "array"
        assert "items" in properties["items"]

        item_props = properties["items"]["items"]["properties"]

        assert "description" in item_props["product_id"]
        assert item_props["product_id"]["description"] == "Test product ID description"

        assert "default" in item_props["quantity"]
        assert item_props["quantity"]["default"] == 1


def test_body_params_edge_cases(complex_fastapi_app: FastAPI):
    """
    Test handling of edge cases for body parameters, such as:
    - Empty or missing descriptions
    - Missing type information
    - Empty properties object
    - Schema without properties
    """
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    order_request_schema = openapi_schema["components"]["schemas"]["OrderRequest"]

    if "description" in order_request_schema["properties"]["customer_id"]:
        del order_request_schema["properties"]["customer_id"]["description"]

    if "type" in order_request_schema["properties"]["notes"]:
        del order_request_schema["properties"]["notes"]["type"]

    item_schema = openapi_schema["components"]["schemas"]["OrderItem"]

    if "properties" in item_schema["properties"]["total"]:
        del item_schema["properties"]["total"]["properties"]

    tools, _ = convert_openapi_to_mcp_tools(openapi_schema)

    create_order_tool = next(tool for tool in tools if tool.name == "create_order")
    properties = create_order_tool.inputSchema["properties"]

    assert "customer_id" in properties
    assert "title" in properties["customer_id"]
    assert properties["customer_id"]["title"] == "customer_id"

    assert "notes" in properties
    assert "type" in properties["notes"]
    assert properties["notes"]["type"] in ["string", "object"]  # Default should be either string or object

    if "items" in properties:
        item_props = properties["items"]["items"]["properties"]
        assert "total" in item_props


def test_parameter_docs_formatting():
    """Test the parameter documentation formatting functions."""
    # Test format_parameter_docs with a mix of required and optional params
    path_params = [
        ("user_id", {"name": "user_id", "schema": {"type": "string"}, "description": "The user ID", "required": True}),
    ]
    query_params = [
        ("limit", {"name": "limit", "schema": {"type": "integer", "default": 10}, "description": "Max results"}),
        ("offset", {"name": "offset", "schema": {"type": "integer", "default": 0}, "description": "Skip results"}),
    ]
    body_params = [
        ("name", {"name": "name", "schema": {"type": "string", "description": "User name"}, "required": True}),
        ("email", {"name": "email", "schema": {"type": "string", "description": "User email"}}),
    ]

    required_props = ["user_id", "name"]

    result = format_all_parameters_docs(path_params, query_params, body_params, required_props)

    # Verify section header
    assert "### Parameters:" in result

    # Verify parameter type sections
    assert "**Path Parameters:**" in result
    assert "**Query Parameters:**" in result
    assert "**Body Parameters:**" in result

    # Verify required params appear with *required* marker
    assert "`user_id` (string) *required*" in result
    assert "The user ID" in result
    assert "`name` (string) *required*" in result
    assert "User name" in result

    # Verify optional params appear with defaults (no *required* marker)
    assert "`limit` (integer)" in result
    assert "*required*" not in result.split("**Query Parameters:**")[1].split("`limit`")[1].split("\n")[0]
    assert "[default: 10]" in result
    assert "`offset` (integer)" in result
    assert "[default: 0]" in result
    assert "`email` (string)" in result


def test_parameter_docs_empty():
    """Test that empty parameter lists return empty string."""
    result = format_all_parameters_docs([], [], [], [])
    assert result == ""


def test_parameter_docs_only_required():
    """Test parameter docs with only required params."""
    path_params = [
        ("id", {"name": "id", "schema": {"type": "integer"}, "required": True}),
    ]
    result = format_all_parameters_docs(path_params, [], [], ["id"])

    assert "**Path Parameters:**" in result
    assert "`id` (integer) *required*" in result


def test_parameter_docs_only_optional():
    """Test parameter docs with only optional params."""
    query_params = [
        ("page", {"name": "page", "schema": {"type": "integer", "default": 1}}),
    ]
    result = format_all_parameters_docs([], query_params, [], [])

    assert "**Query Parameters:**" in result
    assert "`page` (integer)" in result
    assert "*required*" not in result
    assert "[default: 1]" in result


def test_parameter_docs_default_values():
    """Test various default value types are formatted correctly."""
    params = [
        ("str_param", {"name": "str_param", "schema": {"type": "string", "default": "hello"}}),
        ("bool_param", {"name": "bool_param", "schema": {"type": "boolean", "default": False}}),
        ("null_param", {"name": "null_param", "schema": {"type": "string", "default": None}}),
        ("num_param", {"name": "num_param", "schema": {"type": "number", "default": 3.14}}),
    ]
    result = format_all_parameters_docs([], params, [], [])

    assert '[default: "hello"]' in result
    assert "[default: False]" in result
    assert "[default: null]" in result
    assert "[default: 3.14]" in result


def test_tool_description_includes_parameters(complex_fastapi_app: FastAPI):
    """Test that tool descriptions include the Parameters section with parameter type grouping."""
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    tools, _ = convert_openapi_to_mcp_tools(openapi_schema)

    # Find the list_products tool which has many parameters
    list_products_tool = next(tool for tool in tools if tool.name == "list_products")
    description = list_products_tool.description or ""

    # Verify the description includes the Parameters section
    assert "### Parameters:" in description
    assert "**Query Parameters:**" in description

    # Verify that the Responses section still exists and comes after Parameters
    assert "### Responses:" in description
    params_pos = description.find("### Parameters:")
    responses_pos = description.find("### Responses:")
    assert params_pos < responses_pos, "Parameters section should appear before Responses section"


def test_tool_description_structure_order(simple_fastapi_app: FastAPI):
    """Test that tool descriptions follow the correct order: Summary -> Parameters -> Responses."""
    openapi_schema = get_openapi(
        title=simple_fastapi_app.title,
        version=simple_fastapi_app.version,
        openapi_version=simple_fastapi_app.openapi_version,
        description=simple_fastapi_app.description,
        routes=simple_fastapi_app.routes,
    )

    tools, _ = convert_openapi_to_mcp_tools(openapi_schema)

    # Find a tool with a path parameter (get_item has item_id)
    get_item_tool = next(tool for tool in tools if tool.name == "get_item")
    description = get_item_tool.description or ""

    # Verify parameters section exists with type grouping and required markers
    assert "### Parameters:" in description
    assert "**Path Parameters:**" in description
    assert "`item_id`" in description
    assert "*required*" in description

    # Verify section ordering
    params_pos = description.find("### Parameters:")
    responses_pos = description.find("### Responses:")

    # Parameters should come before responses
    assert params_pos < responses_pos


def test_tool_summary_generation():
    """Test the generate_tool_summary function."""
    # Test with summary and description
    result = generate_tool_summary(
        summary="List all items",
        description="Returns a paginated list of all items in the database.",
        method="get",
        path="/items",
        required_count=0,
        optional_count=2,
    )

    assert "**List all items**" in result
    assert "(2 optional params)" in result
    assert "Returns a paginated list" in result

    # Test with required and optional params
    result = generate_tool_summary(
        summary="Create item",
        description="Creates a new item.",
        method="post",
        path="/items",
        required_count=2,
        optional_count=1,
    )

    assert "**Create item**" in result
    assert "(2 required, 1 optional params)" in result

    # Test without summary (fallback to method + path)
    result = generate_tool_summary(
        summary="",
        description="Some description",
        method="delete",
        path="/items/{id}",
        required_count=1,
        optional_count=0,
    )

    assert "**DELETE /items/{id}**" in result
    assert "(1 required params)" in result

    # Test with no params
    result = generate_tool_summary(
        summary="Health check",
        description="",
        method="get",
        path="/health",
        required_count=0,
        optional_count=0,
    )

    assert "**Health check**" in result
    assert "params" not in result


def test_tool_summary_truncates_long_description():
    """Test that long descriptions are truncated in the summary."""
    long_desc = "This is a very long description that goes on and on. " * 10

    result = generate_tool_summary(
        summary="Test endpoint",
        description=long_desc,
        method="get",
        path="/test",
        required_count=0,
        optional_count=0,
    )

    assert "**Test endpoint**" in result
    assert "..." in result
    assert len(result) < len(long_desc)


def test_tool_description_starts_with_summary(complex_fastapi_app: FastAPI):
    """Test that tool descriptions start with a bold summary line."""
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    tools, _ = convert_openapi_to_mcp_tools(openapi_schema)

    for tool in tools:
        description = tool.description or ""
        # Description should start with bold text (summary)
        assert description.startswith("**"), f"Tool {tool.name} description should start with bold summary"
        # Should contain parameter count hint
        assert "params)" in description or "### Parameters:" not in description


def test_required_params_before_optional_in_description(complex_fastapi_app: FastAPI):
    """Test that required parameters (marked with *required*) appear before optional parameters within each group."""
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    tools, _ = convert_openapi_to_mcp_tools(openapi_schema)

    # Find a tool with both required and optional params in the same group
    create_order_tool = next(tool for tool in tools if tool.name == "create_order")
    description = create_order_tool.description or ""

    # Verify that parameter type sections exist
    assert "**Body Parameters:**" in description

    # Within Body Parameters, required params should have *required* marker
    body_section = description.split("**Body Parameters:**")[1].split("### Responses:")[0]

    # Check that required params have the marker
    assert "*required*" in body_section

    # Check that parameters without defaults and that are required have the marker
    # customer_id should be required
    lines = body_section.strip().split("\n")
    required_lines = [line for line in lines if "*required*" in line]
    optional_lines = [line for line in lines if line.startswith("- `") and "*required*" not in line]

    # Required params should appear before optional in the body section
    if required_lines and optional_lines:
        first_required_idx = next(i for i, line in enumerate(lines) if "*required*" in line)
        last_required_idx = len(lines) - 1 - next(
            i for i, line in enumerate(reversed(lines)) if "*required*" in line
        )
        first_optional_idx = next(
            (i for i, line in enumerate(lines) if line.startswith("- `") and "*required*" not in line),
            len(lines),
        )

        # All required params should come before any optional params
        assert last_required_idx < first_optional_idx, (
            f"Required params should come before optional params in body section"
        )


def test_parameter_type_sections_in_description(simple_fastapi_app: FastAPI):
    """Test that parameters are grouped by type (path, query, body) with clear headers."""
    openapi_schema = get_openapi(
        title=simple_fastapi_app.title,
        version=simple_fastapi_app.version,
        openapi_version=simple_fastapi_app.openapi_version,
        description=simple_fastapi_app.description,
        routes=simple_fastapi_app.routes,
    )

    tools, _ = convert_openapi_to_mcp_tools(openapi_schema)

    # Find a tool with path parameter (get_item)
    get_item_tool = next(tool for tool in tools if tool.name == "get_item")
    description = get_item_tool.description or ""

    # Should have Path Parameters section for item_id
    assert "**Path Parameters:**" in description

    # Should have Query Parameters section for include_details
    assert "**Query Parameters:**" in description

    # Verify order: Path Parameters should come before Query Parameters
    path_pos = description.find("**Path Parameters:**")
    query_pos = description.find("**Query Parameters:**")
    assert path_pos < query_pos, "Path Parameters should appear before Query Parameters"


def test_default_param_description_id_patterns():
    """Test default description generation for ID-like parameters."""
    # Simple ID
    assert "identifier" in generate_default_param_description("id", "integer").lower()

    # Entity IDs
    assert "user" in generate_default_param_description("user_id", "integer").lower()
    assert "order" in generate_default_param_description("order_id", "string").lower()
    assert "item" in generate_default_param_description("itemId", "string").lower()


def test_default_param_description_pagination():
    """Test default description generation for pagination parameters."""
    assert "page" in generate_default_param_description("page", "integer").lower()
    assert "number" in generate_default_param_description("limit", "integer").lower()
    assert "skip" in generate_default_param_description("offset", "integer").lower()
    assert "skip" in generate_default_param_description("skip", "integer").lower()


def test_default_param_description_sorting_filtering():
    """Test default description generation for sorting/filtering parameters."""
    assert "sort" in generate_default_param_description("sort_by", "string").lower()
    assert "direction" in generate_default_param_description("order", "string").lower()
    assert "search" in generate_default_param_description("search", "string").lower()
    assert "search" in generate_default_param_description("q", "string").lower()


def test_default_param_description_common_fields():
    """Test default description generation for common field names."""
    assert "name" in generate_default_param_description("name", "string").lower()
    assert "email" in generate_default_param_description("email", "string").lower()
    assert "status" in generate_default_param_description("status", "string").lower()
    assert "tag" in generate_default_param_description("tags", "array").lower()


def test_default_param_description_boolean_flags():
    """Test default description generation for boolean flag parameters."""
    assert "whether" in generate_default_param_description("is_active", "boolean").lower()
    assert "whether" in generate_default_param_description("has_children", "boolean").lower()
    assert "include" in generate_default_param_description("include_details", "boolean").lower()


def test_default_param_description_numeric_ranges():
    """Test default description generation for min/max parameters."""
    assert "minimum" in generate_default_param_description("min_price", "number").lower()
    assert "maximum" in generate_default_param_description("max_count", "integer").lower()


def test_default_param_description_path_params():
    """Test default description generation for path parameters."""
    result = generate_default_param_description("resource_name", "string", "path")
    assert "resource name" in result.lower()
    assert "operate" in result.lower()


def test_default_param_description_unknown():
    """Test that unknown parameter names return empty string."""
    result = generate_default_param_description("xyz_abc_123", "string", "query")
    assert result == ""


def test_params_without_description_get_default(simple_fastapi_app: FastAPI):
    """Test that parameters without descriptions get generated default descriptions."""
    # Create params without descriptions
    path_params = [
        ("item_id", {"name": "item_id", "schema": {"type": "integer"}}),  # No description
    ]
    query_params = [
        ("limit", {"name": "limit", "schema": {"type": "integer", "default": 10}}),  # No description
        ("page", {"name": "page", "schema": {"type": "integer", "default": 1}}),  # No description
    ]

    result = format_all_parameters_docs(path_params, query_params, [], ["item_id"])

    # Should have generated descriptions for known parameter names
    assert "identifier" in result.lower() or "item" in result.lower()
    assert "number of items" in result.lower() or "maximum" in result.lower()
    assert "page" in result.lower()


def test_tool_description_format_consistency(simple_fastapi_app: FastAPI, complex_fastapi_app: FastAPI):
    """Test that all tool descriptions follow a consistent format."""
    for app in [simple_fastapi_app, complex_fastapi_app]:
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )

        tools, _ = convert_openapi_to_mcp_tools(openapi_schema)

        for tool in tools:
            description = tool.description or ""

            # All descriptions should start with bold summary
            assert description.startswith("**"), f"Tool {tool.name} should start with bold summary"

            # All descriptions with parameters should have ### Parameters: section
            if tool.inputSchema.get("properties"):
                assert "### Parameters:" in description, f"Tool {tool.name} should have Parameters section"

            # All descriptions should have ### Responses: section
            assert "### Responses:" in description, f"Tool {tool.name} should have Responses section"

            # Parameters should come before Responses (if both exist)
            if "### Parameters:" in description:
                params_pos = description.find("### Parameters:")
                responses_pos = description.find("### Responses:")
                assert params_pos < responses_pos, f"Tool {tool.name}: Parameters should come before Responses"


def test_tool_description_no_empty_sections(complex_fastapi_app: FastAPI):
    """Test that tool descriptions don't have empty parameter sections."""
    openapi_schema = get_openapi(
        title=complex_fastapi_app.title,
        version=complex_fastapi_app.version,
        openapi_version=complex_fastapi_app.openapi_version,
        description=complex_fastapi_app.description,
        routes=complex_fastapi_app.routes,
    )

    tools, _ = convert_openapi_to_mcp_tools(openapi_schema)

    for tool in tools:
        description = tool.description or ""

        # If a parameter type section exists, it should have content
        for section in ["**Path Parameters:**", "**Query Parameters:**", "**Body Parameters:**"]:
            if section in description:
                # Find the section and check it has content before the next section
                section_start = description.find(section)
                section_content = description[section_start + len(section):]
                # Should have at least one parameter (starts with "- `")
                next_section = min(
                    (section_content.find(s) for s in ["**Path", "**Query", "**Body", "### Responses"]
                     if section_content.find(s) != -1),
                    default=len(section_content)
                )
                content = section_content[:next_section].strip()
                assert "- `" in content, f"Tool {tool.name}: {section} section should have parameters"


def test_contextual_example_values():
    """Test that example generation produces contextual values based on property names."""
    schema = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "email": {"type": "string"},
            "price": {"type": "number"},
            "count": {"type": "integer"},
            "is_active": {"type": "boolean"},
            "created_at": {"type": "string", "format": "date-time"},
        }
    }

    result = generate_example_from_schema(schema)

    # Check contextual values
    assert "abc123" in result["user_id"] or "123" in result["user_id"]
    assert "@" in result["email"]  # Email should have @ symbol
    assert result["price"] == 99.99  # Price should be contextual
    assert result["count"] == 42  # Count should be contextual
    assert result["is_active"] is True  # is_active should be True
    assert "T" in result["created_at"]  # ISO datetime format
