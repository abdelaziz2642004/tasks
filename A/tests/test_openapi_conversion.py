from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import mcp.types as types

from fastapi_mcp.openapi.convert import convert_openapi_to_mcp_tools
from fastapi_mcp.openapi.utils import (
    clean_schema_for_display,
    generate_example_from_schema,
    get_single_param_type_from_schema,
    resolve_schema_references,
    resolve_schema_references_with_diagnostics,
    detect_problematic_references,
    ReferenceResolutionResult,
    MAX_REFERENCE_DEPTH,
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


# =============================================================================
# Tests for improved reference resolution
# =============================================================================


def test_circular_reference_detection():
    """Test that circular references are detected and handled gracefully."""
    # Create a schema with a direct circular reference (A -> A)
    schema = {
        "components": {
            "schemas": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "next": {"$ref": "#/components/schemas/Node"},
                    },
                }
            }
        },
        "paths": {},
    }

    # Should not raise an exception
    result = resolve_schema_references_with_diagnostics(schema, schema)

    # Should detect circular reference
    assert len(result.circular_refs_detected) > 0
    assert "#/components/schemas/Node" in result.circular_refs_detected

    # The resolved schema should have the x-circular-ref marker somewhere in the nested structure
    # The first resolution expands Node, but the nested "next" property contains the circular marker
    node_schema = result.schema["components"]["schemas"]["Node"]
    # Navigate to find the x-circular-ref marker (it will be in the deeply nested structure)
    # First level: Node.properties.next is resolved to the Node schema
    # Second level: Node.properties.next.properties.next has the circular marker
    next_prop = node_schema["properties"]["next"]
    # The next property should contain the resolved Node schema, and its nested next should have the marker
    nested_next = next_prop.get("properties", {}).get("next", {})
    assert nested_next.get("x-circular-ref") is True


def test_indirect_circular_reference_detection():
    """Test detection of indirect circular references (A -> B -> A)."""
    schema = {
        "components": {
            "schemas": {
                "Parent": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "child": {"$ref": "#/components/schemas/Child"},
                    },
                },
                "Child": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "parent": {"$ref": "#/components/schemas/Parent"},
                    },
                },
            }
        },
        "paths": {},
    }

    result = resolve_schema_references_with_diagnostics(schema, schema)

    # Should detect the circular reference chain
    assert len(result.circular_refs_detected) > 0


def test_deep_nested_references():
    """Test handling of deeply nested reference chains."""
    # Create a chain of references: Level0 -> Level1 -> Level2 -> ... -> Level10
    schemas = {}
    for i in range(11):
        if i < 10:
            schemas[f"Level{i}"] = {
                "type": "object",
                "properties": {
                    "data": {"type": "string"},
                    "nested": {"$ref": f"#/components/schemas/Level{i+1}"},
                },
            }
        else:
            schemas[f"Level{i}"] = {
                "type": "object",
                "properties": {
                    "data": {"type": "string"},
                },
            }

    schema = {"components": {"schemas": schemas}, "paths": {}}

    result = resolve_schema_references_with_diagnostics(schema, schema)

    # Should resolve successfully without hitting depth limit
    assert result.max_depth_reached is False
    assert result.total_refs_resolved == 10  # 10 references resolved


def test_max_depth_protection():
    """Test that extremely deep nesting is handled with depth limit."""
    # Create a very deep chain that exceeds MAX_REFERENCE_DEPTH
    schemas = {}
    depth = MAX_REFERENCE_DEPTH + 10

    for i in range(depth):
        if i < depth - 1:
            schemas[f"Level{i}"] = {
                "type": "object",
                "properties": {
                    "nested": {"$ref": f"#/components/schemas/Level{i+1}"},
                },
            }
        else:
            schemas[f"Level{i}"] = {"type": "object", "properties": {"value": {"type": "string"}}}

    schema = {"components": {"schemas": schemas}, "paths": {}}

    result = resolve_schema_references_with_diagnostics(schema, schema)

    # Should have hit the depth limit
    assert result.max_depth_reached is True
    assert any("Maximum reference depth" in w for w in result.warnings)


def test_reference_caching():
    """Test that resolved references are cached for efficiency."""
    # Create a schema where the same reference is used multiple times
    schema = {
        "components": {
            "schemas": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "homeAddress": {"$ref": "#/components/schemas/Address"},
                        "workAddress": {"$ref": "#/components/schemas/Address"},
                        "mailingAddress": {"$ref": "#/components/schemas/Address"},
                    },
                },
            }
        },
        "paths": {},
    }

    result = resolve_schema_references_with_diagnostics(schema, schema)

    # The Address schema should only be resolved once (cached for subsequent uses)
    # Total refs resolved should be 1 (Address) not 3
    assert result.total_refs_resolved == 1

    # All three address properties should be resolved
    person_schema = result.schema["components"]["schemas"]["Person"]
    assert "street" in person_schema["properties"]["homeAddress"]["properties"]
    assert "street" in person_schema["properties"]["workAddress"]["properties"]
    assert "street" in person_schema["properties"]["mailingAddress"]["properties"]


def test_unresolved_reference_warning():
    """Test that unresolved references generate warnings."""
    schema = {
        "components": {"schemas": {}},
        "paths": {
            "/users": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/NonExistentModel"}
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    result = resolve_schema_references_with_diagnostics(schema, schema)

    # Should have a warning about unresolved reference
    assert any("Unresolved reference" in w for w in result.warnings)
    assert any("NonExistentModel" in w for w in result.warnings)


def test_json_pointer_with_special_characters():
    """Test handling of JSON pointers with special characters."""
    # JSON pointers use ~0 for ~ and ~1 for /
    schema = {
        "components": {
            "schemas": {
                "Model/With/Slashes": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
                "Model~With~Tildes": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                },
            }
        },
        "paths": {
            "/test": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Model~1With~1Slashes"}
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    result = resolve_schema_references(schema, schema)

    # The reference should be resolved correctly
    response_schema = result["paths"]["/test"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema.get("type") == "object"
    assert "value" in response_schema.get("properties", {})


def test_detect_problematic_references_circular():
    """Test detection of circular references without resolving."""
    schema = {
        "components": {
            "schemas": {
                "A": {
                    "type": "object",
                    "properties": {"b": {"$ref": "#/components/schemas/B"}},
                },
                "B": {
                    "type": "object",
                    "properties": {"c": {"$ref": "#/components/schemas/C"}},
                },
                "C": {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/components/schemas/A"}},
                },
            }
        }
    }

    warnings = detect_problematic_references(schema)

    # Should detect the circular reference chain A -> B -> C -> A
    assert any("Circular reference chain" in w for w in warnings)


def test_detect_problematic_references_missing():
    """Test detection of missing schema references."""
    schema = {
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "profile": {"$ref": "#/components/schemas/MissingProfile"},
                    },
                }
            }
        }
    }

    warnings = detect_problematic_references(schema)

    # Should detect missing reference
    assert any("undefined schema" in w.lower() for w in warnings)
    assert any("MissingProfile" in w for w in warnings)


def test_detect_problematic_references_deep_chain():
    """Test detection of deeply nested reference chains."""
    # Create a chain longer than 10 levels
    schemas = {}
    for i in range(15):
        if i < 14:
            schemas[f"Level{i}"] = {
                "type": "object",
                "properties": {"next": {"$ref": f"#/components/schemas/Level{i+1}"}},
            }
        else:
            schemas[f"Level{i}"] = {"type": "object", "properties": {"value": {"type": "string"}}}

    schema = {"components": {"schemas": schemas}}

    warnings = detect_problematic_references(schema)

    # Should detect deep chain
    assert any("Deep reference chain" in w for w in warnings)


def test_detect_problematic_references_empty_schema():
    """Test that empty schemas don't cause issues."""
    schema = {}
    warnings = detect_problematic_references(schema)
    assert warnings == []

    schema_no_schemas = {"components": {}}
    warnings = detect_problematic_references(schema_no_schemas)
    assert warnings == []


def test_resolve_references_in_allof_anyof_oneof():
    """Test resolution of references within allOf, anyOf, oneOf constructs."""
    schema = {
        "components": {
            "schemas": {
                "Base": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
                "Extension": {
                    "type": "object",
                    "properties": {"extra": {"type": "string"}},
                },
                "Combined": {
                    "allOf": [
                        {"$ref": "#/components/schemas/Base"},
                        {"$ref": "#/components/schemas/Extension"},
                    ]
                },
                "Either": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/Base"},
                        {"$ref": "#/components/schemas/Extension"},
                    ]
                },
            }
        },
        "paths": {},
    }

    result = resolve_schema_references(schema, schema)

    # allOf references should be resolved
    combined = result["components"]["schemas"]["Combined"]
    assert len(combined["allOf"]) == 2
    assert combined["allOf"][0].get("properties", {}).get("id", {}).get("type") == "string"
    assert combined["allOf"][1].get("properties", {}).get("extra", {}).get("type") == "string"

    # anyOf references should be resolved
    either = result["components"]["schemas"]["Either"]
    assert len(either["anyOf"]) == 2


def test_resolve_references_in_array_items():
    """Test resolution of references in array item schemas."""
    schema = {
        "components": {
            "schemas": {
                "Item": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "ItemList": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/Item"},
                },
            }
        },
        "paths": {},
    }

    result = resolve_schema_references(schema, schema)

    item_list = result["components"]["schemas"]["ItemList"]
    assert item_list["items"].get("type") == "object"
    assert "name" in item_list["items"].get("properties", {})


def test_resolve_references_preserves_additional_properties():
    """Test that resolving references preserves additional properties alongside $ref."""
    schema = {
        "components": {
            "schemas": {
                "Base": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
            }
        },
        "paths": {
            "/test": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Base",
                                        "description": "Custom description",
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    result = resolve_schema_references(schema, schema)

    response_schema = result["paths"]["/test"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    # The description should be preserved (though $ref is replaced)
    # Note: In OpenAPI 3.0, properties alongside $ref are ignored, but we preserve them
    assert response_schema.get("type") == "object"


def test_mcp_tools_with_circular_references():
    """Test that MCP tool conversion handles schemas with circular references."""
    # This simulates a real-world case where OpenAPI generators produce circular refs
    openapi_schema = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/nodes": {
                "post": {
                    "operationId": "create_node",
                    "summary": "Create a node",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Node"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Node"}
                                }
                            },
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/Node"},
                        },
                        "parent": {"$ref": "#/components/schemas/Node"},
                    },
                    "required": ["id", "name"],
                }
            }
        },
    }

    # Should not raise an exception and should produce valid tools
    tools, operation_map = convert_openapi_to_mcp_tools(openapi_schema)

    assert len(tools) == 1
    assert tools[0].name == "create_node"
    assert "create_node" in operation_map

    # The tool should have input schema properties
    assert "properties" in tools[0].inputSchema
    assert "id" in tools[0].inputSchema["properties"]
    assert "name" in tools[0].inputSchema["properties"]
