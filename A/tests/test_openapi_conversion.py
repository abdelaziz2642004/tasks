from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import mcp.types as types

import pytest
from fastapi_mcp.openapi.convert import convert_openapi_to_mcp_tools
from fastapi_mcp.openapi.utils import (
    clean_schema_for_display,
    generate_example_from_schema,
    get_single_param_type_from_schema,
    resolve_schema_references,
    resolve_schema_references_with_details,
    resolve_schema_references_strict,
    validate_resolved_schema,
    analyze_schema_references,
    ReferenceResolutionResult,
    SchemaAnalysisResult,
    UnresolvedReferenceError,
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


# Tests for improved reference resolution


def test_circular_reference_detection():
    """Test that circular references are detected and handled gracefully."""
    # Schema with circular reference: A -> B -> A
    schema = {
        "components": {
            "schemas": {
                "NodeA": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "child": {"$ref": "#/components/schemas/NodeB"},
                    },
                },
                "NodeB": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                        "parent": {"$ref": "#/components/schemas/NodeA"},
                    },
                },
            }
        },
        "paths": {
            "/test": {
                "get": {
                    "operationId": "test_op",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/NodeA"}
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    # Should not raise an error
    result = resolve_schema_references_with_details(schema, schema)

    assert isinstance(result, ReferenceResolutionResult)
    assert len(result.circular_refs) > 0
    assert "#/components/schemas/NodeA" in result.circular_refs or "#/components/schemas/NodeB" in result.circular_refs


def test_self_referencing_schema():
    """Test schema that references itself (common in tree structures)."""
    schema = {
        "components": {
            "schemas": {
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/TreeNode"},
                        },
                    },
                }
            }
        }
    }

    result = resolve_schema_references_with_details(schema, schema)

    assert "#/components/schemas/TreeNode" in result.circular_refs


def test_deeply_nested_references():
    """Test handling of deeply nested references."""
    # Create a chain: A -> B -> C -> D -> E
    schema = {
        "components": {
            "schemas": {
                "LevelA": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/LevelB"}},
                },
                "LevelB": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/LevelC"}},
                },
                "LevelC": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/LevelD"}},
                },
                "LevelD": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/LevelE"}},
                },
                "LevelE": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            }
        }
    }

    # Should resolve without issues
    result = resolve_schema_references(schema, schema)

    # Verify the chain was resolved
    assert "components" in result
    level_a = result["components"]["schemas"]["LevelA"]
    assert "properties" in level_a
    assert "next" in level_a["properties"]


def test_unresolved_reference_warning():
    """Test that unresolved references generate warnings."""
    schema = {
        "paths": {
            "/test": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/NonExistent"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    result = resolve_schema_references_with_details(schema, schema)

    assert len(result.warnings) > 0
    assert any("NonExistent" in w for w in result.warnings)


def test_analyze_schema_circular_refs():
    """Test schema analysis detects circular references."""
    schema = {
        "components": {
            "schemas": {
                "Parent": {
                    "type": "object",
                    "properties": {
                        "child": {"$ref": "#/components/schemas/Child"},
                    },
                },
                "Child": {
                    "type": "object",
                    "properties": {
                        "parent": {"$ref": "#/components/schemas/Parent"},
                    },
                },
            }
        }
    }

    analysis = analyze_schema_references(schema)

    assert isinstance(analysis, SchemaAnalysisResult)
    assert len(analysis.circular_refs) > 0
    assert analysis.has_issues


def test_analyze_schema_external_refs():
    """Test schema analysis detects external references."""
    schema = {
        "paths": {
            "/test": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "https://example.com/schema.json#/Model"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    analysis = analyze_schema_references(schema)

    assert len(analysis.external_refs) == 1
    assert "https://example.com/schema.json#/Model" in analysis.external_refs
    assert analysis.has_issues


def test_analyze_schema_unresolved_refs():
    """Test schema analysis detects unresolved references."""
    schema = {
        "components": {"schemas": {}},
        "paths": {
            "/test": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Missing"}
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    analysis = analyze_schema_references(schema)

    assert len(analysis.unresolved_refs) == 1
    assert "#/components/schemas/Missing" in analysis.unresolved_refs
    assert analysis.has_issues


def test_analyze_schema_no_issues():
    """Test schema analysis with no issues."""
    schema = {
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                }
            }
        },
        "paths": {
            "/users": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/User"}
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    analysis = analyze_schema_references(schema)

    assert len(analysis.all_refs) == 1
    assert len(analysis.circular_refs) == 0
    assert len(analysis.unresolved_refs) == 0
    assert len(analysis.external_refs) == 0
    assert not analysis.has_issues


def test_json_pointer_special_characters():
    """Test JSON pointer resolution with special characters."""
    schema = {
        "components": {
            "schemas": {
                "Model/With/Slashes": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                }
            }
        },
        "paths": {
            "/test": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    # ~1 encodes / in JSON pointer (RFC 6901)
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

    # The reference should be resolved
    response_schema = result["paths"]["/test"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema.get("type") == "object"
    assert "value" in response_schema.get("properties", {})


def test_reference_with_additional_properties():
    """Test that additional properties alongside $ref are preserved."""
    schema = {
        "components": {
            "schemas": {
                "BaseModel": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                }
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
                                        "$ref": "#/components/schemas/BaseModel",
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
    assert response_schema.get("type") == "object"
    assert response_schema.get("description") == "Custom description"


def test_reference_caching():
    """Test that reference resolution uses caching efficiently."""
    # Schema where the same reference is used multiple times
    schema = {
        "components": {
            "schemas": {
                "SharedModel": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                }
            }
        },
        "paths": {
            "/a": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/SharedModel"}}
                            }
                        }
                    }
                }
            },
            "/b": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/SharedModel"}}
                            }
                        }
                    }
                }
            },
            "/c": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/SharedModel"}}
                            }
                        }
                    }
                }
            },
        },
    }

    result = resolve_schema_references_with_details(schema, schema)

    # All three should be resolved
    for path in ["/a", "/b", "/c"]:
        response_schema = result.schema["paths"][path]["get"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]
        assert response_schema.get("type") == "object"

    # Verify caching is working - should have cache hits for repeated references
    # First reference resolves, subsequent ones should hit cache
    assert result.cache_hits >= 2  # At least 2 cache hits for 3 uses of same ref


def test_complex_nested_refs():
    """Test complex nested reference structures."""
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
                        "address": {"$ref": "#/components/schemas/Address"},
                    },
                },
                "Company": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "ceo": {"$ref": "#/components/schemas/Person"},
                        "headquarters": {"$ref": "#/components/schemas/Address"},
                    },
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    company = result["components"]["schemas"]["Company"]
    assert "properties" in company
    assert company["properties"]["ceo"].get("type") == "object"
    assert "address" in company["properties"]["ceo"]["properties"]
    assert company["properties"]["ceo"]["properties"]["address"].get("type") == "object"
    assert company["properties"]["headquarters"].get("type") == "object"


def test_allof_with_refs():
    """Test allOf with references."""
    schema = {
        "components": {
            "schemas": {
                "BaseModel": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
                "ExtendedModel": {
                    "allOf": [
                        {"$ref": "#/components/schemas/BaseModel"},
                        {
                            "type": "object",
                            "properties": {"extra": {"type": "string"}},
                        },
                    ]
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    extended = result["components"]["schemas"]["ExtendedModel"]
    assert "allOf" in extended
    assert len(extended["allOf"]) == 2
    # First item should be resolved
    assert extended["allOf"][0].get("type") == "object"
    assert "id" in extended["allOf"][0].get("properties", {})


def test_array_of_refs():
    """Test array with items as reference."""
    schema = {
        "components": {
            "schemas": {
                "Item": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            }
        },
        "paths": {
            "/items": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Item"},
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

    response_schema = result["paths"]["/items"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema.get("type") == "array"
    assert response_schema["items"].get("type") == "object"
    assert "name" in response_schema["items"].get("properties", {})


# Tests for validation and metadata preservation


def test_validate_resolved_schema_success():
    """Test that validation passes for fully resolved schemas."""
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
        },
    }

    is_valid, unresolved = validate_resolved_schema(schema)

    assert is_valid
    assert len(unresolved) == 0


def test_validate_resolved_schema_with_unresolved_refs():
    """Test that validation detects unresolved references."""
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "related": {"$ref": "#/components/schemas/Missing"},
        },
    }

    is_valid, unresolved = validate_resolved_schema(schema)

    assert not is_valid
    assert "#/components/schemas/Missing" in unresolved


def test_validate_resolved_schema_ignores_circular_refs():
    """Test that validation ignores properly marked circular references."""
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "parent": {"$ref": "#/components/schemas/Self", "_circular": True},
        },
    }

    is_valid, unresolved = validate_resolved_schema(schema)

    # Circular refs are expected and should not fail validation
    assert is_valid
    assert len(unresolved) == 0


def test_resolve_schema_references_strict_success():
    """Test strict resolution succeeds with valid schema."""
    schema = {
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                }
            }
        },
        "paths": {
            "/users": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/User"}}
                            }
                        }
                    }
                }
            }
        },
    }

    result = resolve_schema_references_strict(schema, schema)

    response_schema = result["paths"]["/users"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema.get("type") == "object"


def test_resolve_schema_references_strict_raises_on_unresolved():
    """Test strict resolution raises error on unresolved references."""
    schema = {
        "paths": {
            "/test": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/Missing"}}
                            }
                        }
                    }
                }
            }
        }
    }

    with pytest.raises(UnresolvedReferenceError) as exc_info:
        resolve_schema_references_strict(schema, schema)

    assert "#/components/schemas/Missing" in exc_info.value.unresolved_refs
    assert "Missing" in str(exc_info.value)


def test_metadata_preservation_description():
    """Test that description metadata is preserved from reference site."""
    schema = {
        "components": {
            "schemas": {
                "BaseModel": {
                    "type": "object",
                    "description": "Base model description",
                    "properties": {"id": {"type": "integer"}},
                }
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
                                        "$ref": "#/components/schemas/BaseModel",
                                        "description": "Overridden description at reference site",
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
    # Description from reference site should take priority
    assert response_schema.get("description") == "Overridden description at reference site"
    assert response_schema.get("type") == "object"


def test_metadata_preservation_title():
    """Test that title metadata is preserved from reference site."""
    schema = {
        "components": {
            "schemas": {
                "BaseModel": {
                    "type": "object",
                    "title": "BaseModelTitle",
                    "properties": {"id": {"type": "integer"}},
                }
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
                                        "$ref": "#/components/schemas/BaseModel",
                                        "title": "CustomTitle",
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
    assert response_schema.get("title") == "CustomTitle"


def test_metadata_preservation_examples():
    """Test that examples metadata is preserved from reference site."""
    schema = {
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "example": {"name": "Default User"},
                }
            }
        },
        "paths": {
            "/users": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/User",
                                        "example": {"name": "Custom Example"},
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

    response_schema = result["paths"]["/users"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema.get("example") == {"name": "Custom Example"}


def test_resolution_result_properties():
    """Test ReferenceResolutionResult properties work correctly."""
    schema = {
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                }
            }
        },
        "paths": {
            "/users": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/User"}}
                            }
                        }
                    }
                }
            }
        },
    }

    result = resolve_schema_references_with_details(schema, schema)

    assert result.is_fully_resolved
    assert not result.has_unresolved_refs
    assert len(result.unresolved_refs) == 0


def test_resolution_result_with_unresolved():
    """Test ReferenceResolutionResult with unresolved references."""
    schema = {
        "paths": {
            "/test": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/Missing"}}
                            }
                        }
                    }
                }
            }
        }
    }

    result = resolve_schema_references_with_details(schema, schema)

    assert not result.is_fully_resolved
    assert result.has_unresolved_refs
    assert "#/components/schemas/Missing" in result.unresolved_refs


def test_nested_unresolved_refs_validation():
    """Test validation catches deeply nested unresolved references."""
    schema = {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/DeepMissing"},
                    }
                },
            }
        },
    }

    is_valid, unresolved = validate_resolved_schema(schema)

    assert not is_valid
    assert "#/components/schemas/DeepMissing" in unresolved


# Complex edge case tests


def test_very_deeply_nested_references():
    """Test resolution of very deeply nested references (10+ levels)."""
    # Create a chain of 10 levels deep
    schema = {
        "components": {
            "schemas": {
                "Level1": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Level2"}}},
                "Level2": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Level3"}}},
                "Level3": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Level4"}}},
                "Level4": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Level5"}}},
                "Level5": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Level6"}}},
                "Level6": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Level7"}}},
                "Level7": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Level8"}}},
                "Level8": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Level9"}}},
                "Level9": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Level10"}}},
                "Level10": {"type": "object", "properties": {"value": {"type": "string"}}},
            }
        },
        "paths": {
            "/deep": {
                "get": {
                    "operationId": "get_deep",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/Level1"}}
                            }
                        }
                    },
                }
            }
        },
    }

    result = resolve_schema_references_with_details(schema, schema)

    # Should resolve successfully
    assert result.is_fully_resolved
    assert len(result.circular_refs) == 0

    # Verify the chain was properly resolved
    level1 = result.schema["components"]["schemas"]["Level1"]
    assert level1["properties"]["next"]["type"] == "object"


def test_references_in_nested_arrays():
    """Test references within nested array structures."""
    schema = {
        "components": {
            "schemas": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                },
                "Container": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/Item"},
                            },
                        }
                    },
                },
            }
        },
        "paths": {
            "/containers": {
                "get": {
                    "operationId": "get_containers",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/Container"}}
                            }
                        }
                    },
                }
            }
        },
    }

    result = resolve_schema_references(schema, schema)

    # Verify nested array items are resolved
    container = result["components"]["schemas"]["Container"]
    inner_items = container["properties"]["items"]["items"]["items"]
    assert inner_items["type"] == "object"
    assert "id" in inner_items["properties"]
    assert "name" in inner_items["properties"]


def test_allof_multiple_refs():
    """Test allOf with multiple references."""
    schema = {
        "components": {
            "schemas": {
                "Identifiable": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "format": "uuid"}},
                    "required": ["id"],
                },
                "Timestamped": {
                    "type": "object",
                    "properties": {
                        "created_at": {"type": "string", "format": "date-time"},
                        "updated_at": {"type": "string", "format": "date-time"},
                    },
                },
                "Named": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "Entity": {
                    "allOf": [
                        {"$ref": "#/components/schemas/Identifiable"},
                        {"$ref": "#/components/schemas/Timestamped"},
                        {"$ref": "#/components/schemas/Named"},
                        {"type": "object", "properties": {"description": {"type": "string"}}},
                    ]
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    entity = result["components"]["schemas"]["Entity"]
    assert "allOf" in entity
    assert len(entity["allOf"]) == 4

    # Verify each allOf item is resolved
    assert entity["allOf"][0]["type"] == "object"
    assert "id" in entity["allOf"][0]["properties"]
    assert entity["allOf"][1]["type"] == "object"
    assert "created_at" in entity["allOf"][1]["properties"]
    assert entity["allOf"][2]["type"] == "object"
    assert "name" in entity["allOf"][2]["properties"]


def test_oneof_with_refs():
    """Test oneOf with references."""
    schema = {
        "components": {
            "schemas": {
                "Cat": {
                    "type": "object",
                    "properties": {
                        "meow": {"type": "boolean"},
                        "whiskers": {"type": "integer"},
                    },
                },
                "Dog": {
                    "type": "object",
                    "properties": {
                        "bark": {"type": "boolean"},
                        "tail_wagging": {"type": "boolean"},
                    },
                },
                "Pet": {
                    "oneOf": [
                        {"$ref": "#/components/schemas/Cat"},
                        {"$ref": "#/components/schemas/Dog"},
                    ],
                    "discriminator": {"propertyName": "pet_type"},
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    pet = result["components"]["schemas"]["Pet"]
    assert "oneOf" in pet
    assert len(pet["oneOf"]) == 2

    # Verify both options are resolved
    assert pet["oneOf"][0]["type"] == "object"
    assert "meow" in pet["oneOf"][0]["properties"]
    assert pet["oneOf"][1]["type"] == "object"
    assert "bark" in pet["oneOf"][1]["properties"]


def test_anyof_with_refs():
    """Test anyOf with references."""
    schema = {
        "components": {
            "schemas": {
                "StringId": {"type": "string", "minLength": 1},
                "IntegerId": {"type": "integer", "minimum": 1},
                "UuidId": {"type": "string", "format": "uuid"},
                "FlexibleId": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/StringId"},
                        {"$ref": "#/components/schemas/IntegerId"},
                        {"$ref": "#/components/schemas/UuidId"},
                    ]
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    flexible_id = result["components"]["schemas"]["FlexibleId"]
    assert "anyOf" in flexible_id
    assert len(flexible_id["anyOf"]) == 3

    # Verify all options are resolved
    assert flexible_id["anyOf"][0]["type"] == "string"
    assert flexible_id["anyOf"][1]["type"] == "integer"
    assert flexible_id["anyOf"][2]["type"] == "string"
    assert flexible_id["anyOf"][2]["format"] == "uuid"


def test_nested_composition_refs():
    """Test nested allOf/oneOf/anyOf with references."""
    schema = {
        "components": {
            "schemas": {
                "Base": {"type": "object", "properties": {"base_field": {"type": "string"}}},
                "OptionA": {"type": "object", "properties": {"option_a": {"type": "boolean"}}},
                "OptionB": {"type": "object", "properties": {"option_b": {"type": "integer"}}},
                "Complex": {
                    "allOf": [
                        {"$ref": "#/components/schemas/Base"},
                        {
                            "type": "object",
                            "properties": {
                                "variant": {
                                    "oneOf": [
                                        {"$ref": "#/components/schemas/OptionA"},
                                        {"$ref": "#/components/schemas/OptionB"},
                                    ]
                                }
                            },
                        },
                    ]
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    complex_schema = result["components"]["schemas"]["Complex"]
    assert "allOf" in complex_schema

    # Verify nested oneOf is resolved
    variant = complex_schema["allOf"][1]["properties"]["variant"]
    assert "oneOf" in variant
    assert variant["oneOf"][0]["type"] == "object"
    assert "option_a" in variant["oneOf"][0]["properties"]


def test_mixed_local_and_external_refs():
    """Test schema with both local and external references."""
    schema = {
        "components": {
            "schemas": {
                "LocalModel": {
                    "type": "object",
                    "properties": {"local_field": {"type": "string"}},
                }
            }
        },
        "paths": {
            "/mixed": {
                "get": {
                    "operationId": "get_mixed",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "local": {"$ref": "#/components/schemas/LocalModel"},
                                            "external": {"$ref": "https://example.com/schemas/External.json"},
                                        },
                                    }
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    result = resolve_schema_references_with_details(schema, schema)

    # Local ref should be resolved, external should remain
    response_schema = result.schema["paths"]["/mixed"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]

    # Local reference should be resolved
    assert response_schema["properties"]["local"]["type"] == "object"
    assert "local_field" in response_schema["properties"]["local"]["properties"]

    # External reference should remain unresolved
    assert "$ref" in response_schema["properties"]["external"]
    assert "https://example.com" in response_schema["properties"]["external"]["$ref"]

    # Should have unresolved refs
    assert result.has_unresolved_refs
    assert "https://example.com/schemas/External.json" in result.unresolved_refs


def test_refs_in_additional_properties():
    """Test references in additionalProperties."""
    schema = {
        "components": {
            "schemas": {
                "Value": {"type": "object", "properties": {"data": {"type": "string"}}},
                "Dictionary": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/components/schemas/Value"},
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    dictionary = result["components"]["schemas"]["Dictionary"]
    assert dictionary["additionalProperties"]["type"] == "object"
    assert "data" in dictionary["additionalProperties"]["properties"]


def test_refs_in_pattern_properties():
    """Test references in patternProperties."""
    schema = {
        "components": {
            "schemas": {
                "MetricValue": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                },
                "Metrics": {
                    "type": "object",
                    "patternProperties": {"^metric_.*$": {"$ref": "#/components/schemas/MetricValue"}},
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    metrics = result["components"]["schemas"]["Metrics"]
    pattern_schema = metrics["patternProperties"]["^metric_.*$"]
    assert pattern_schema["type"] == "object"
    assert "value" in pattern_schema["properties"]


def test_mcp_tool_from_complex_schema():
    """Test that complex resolved schemas produce correct MCP tool definitions."""
    schema = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "components": {
            "schemas": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string", "description": "Street address"},
                        "city": {"type": "string", "description": "City name"},
                        "zip": {"type": "string", "pattern": "^[0-9]{5}$"},
                    },
                    "required": ["street", "city"],
                },
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full name"},
                        "age": {"type": "integer", "minimum": 0},
                        "address": {"$ref": "#/components/schemas/Address"},
                    },
                    "required": ["name"],
                },
            }
        },
        "paths": {
            "/persons": {
                "post": {
                    "operationId": "create_person",
                    "summary": "Create a new person",
                    "description": "Creates a person with optional address",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Person"}}},
                    },
                    "responses": {
                        "201": {
                            "description": "Person created",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Person"}}},
                        }
                    },
                }
            }
        },
    }

    tools, operation_map = convert_openapi_to_mcp_tools(schema)

    assert len(tools) == 1
    tool = tools[0]

    assert tool.name == "create_person"
    assert "Create a new person" in tool.description

    # Check input schema has resolved properties
    input_schema = tool.inputSchema
    assert "properties" in input_schema
    assert "name" in input_schema["properties"]
    assert "age" in input_schema["properties"]
    assert "address" in input_schema["properties"]

    # The address property should have resolved nested properties
    address_schema = input_schema["properties"]["address"]
    assert address_schema["type"] == "object"
    assert "street" in address_schema["properties"]
    assert "city" in address_schema["properties"]

    # Check required fields are preserved
    assert "name" in input_schema.get("required", [])


def test_mcp_tool_preserves_descriptions():
    """Test that descriptions from schemas are preserved in MCP tools."""
    schema = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "components": {
            "schemas": {
                "CreateRequest": {
                    "type": "object",
                    "description": "Request to create an item",
                    "properties": {
                        "name": {"type": "string", "description": "The item name"},
                        "quantity": {"type": "integer", "description": "Number of items", "default": 1},
                    },
                    "required": ["name"],
                }
            }
        },
        "paths": {
            "/items": {
                "post": {
                    "operationId": "create_item",
                    "summary": "Create an item",
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CreateRequest"}}}
                    },
                    "responses": {"200": {"description": "Success"}},
                }
            }
        },
    }

    tools, _ = convert_openapi_to_mcp_tools(schema)

    tool = tools[0]
    props = tool.inputSchema["properties"]

    # Verify descriptions are preserved
    assert props["name"]["description"] == "The item name"
    assert props["quantity"]["description"] == "Number of items"

    # Verify default is preserved
    assert props["quantity"]["default"] == 1


def test_mcp_tool_with_circular_refs():
    """Test that MCP tools handle circular references gracefully."""
    schema = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "components": {
            "schemas": {
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/TreeNode"},
                        },
                    },
                }
            }
        },
        "paths": {
            "/trees": {
                "post": {
                    "operationId": "create_tree",
                    "summary": "Create a tree",
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TreeNode"}}}
                    },
                    "responses": {"200": {"description": "Success"}},
                }
            }
        },
    }

    # Should not raise an error
    tools, _ = convert_openapi_to_mcp_tools(schema)

    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "create_tree"
    assert "value" in tool.inputSchema["properties"]
    assert "children" in tool.inputSchema["properties"]


def test_no_information_loss_during_resolution():
    """Test that no information is lost during schema resolution."""
    schema = {
        "components": {
            "schemas": {
                "FullModel": {
                    "type": "object",
                    "title": "Full Model",
                    "description": "A model with all fields",
                    "properties": {
                        "id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "Unique identifier",
                            "example": "123e4567-e89b-12d3-a456-426614174000",
                        },
                        "count": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "default": 10,
                            "description": "Item count",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": 1,
                            "maxItems": 10,
                            "uniqueItems": True,
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Key-value metadata",
                        },
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                    "example": {"id": "test-id", "count": 5, "tags": ["a", "b"]},
                }
            }
        },
        "paths": {
            "/models": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/FullModel"}}
                            }
                        }
                    }
                }
            }
        },
    }

    result = resolve_schema_references(schema, schema)

    resolved_model = result["components"]["schemas"]["FullModel"]

    # Verify all fields are preserved
    assert resolved_model["type"] == "object"
    assert resolved_model["title"] == "Full Model"
    assert resolved_model["description"] == "A model with all fields"
    assert resolved_model["required"] == ["id"]
    assert resolved_model["additionalProperties"] is False
    assert "example" in resolved_model

    # Verify property details
    id_prop = resolved_model["properties"]["id"]
    assert id_prop["type"] == "string"
    assert id_prop["format"] == "uuid"
    assert id_prop["description"] == "Unique identifier"
    assert "example" in id_prop

    count_prop = resolved_model["properties"]["count"]
    assert count_prop["minimum"] == 0
    assert count_prop["maximum"] == 100
    assert count_prop["default"] == 10

    tags_prop = resolved_model["properties"]["tags"]
    assert tags_prop["minItems"] == 1
    assert tags_prop["maxItems"] == 10
    assert tags_prop["uniqueItems"] is True
    assert tags_prop["items"]["minLength"] == 1


def test_refs_preserve_schema_extensions():
    """Test that x- extension fields are preserved during resolution."""
    schema = {
        "components": {
            "schemas": {
                "ExtendedModel": {
                    "type": "object",
                    "x-custom-field": "custom-value",
                    "x-internal": True,
                    "properties": {
                        "field": {
                            "type": "string",
                            "x-field-metadata": {"key": "value"},
                        }
                    },
                }
            }
        },
        "paths": {
            "/extended": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/ExtendedModel"}}
                            }
                        }
                    }
                }
            }
        },
    }

    result = resolve_schema_references(schema, schema)

    response_schema = result["paths"]["/extended"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]

    # Verify extensions are preserved
    assert response_schema["x-custom-field"] == "custom-value"
    assert response_schema["x-internal"] is True
    assert response_schema["properties"]["field"]["x-field-metadata"] == {"key": "value"}


def test_multiple_refs_to_same_schema_independence():
    """Test that multiple references to the same schema are independent copies."""
    schema = {
        "components": {
            "schemas": {
                "Shared": {"type": "object", "properties": {"value": {"type": "string"}}}
            }
        },
        "paths": {
            "/a": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Shared",
                                        "description": "Description A",
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/b": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Shared",
                                        "description": "Description B",
                                    }
                                }
                            }
                        }
                    }
                }
            },
        },
    }

    result = resolve_schema_references(schema, schema)

    schema_a = result["paths"]["/a"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    schema_b = result["paths"]["/b"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]

    # Both should have the base type
    assert schema_a["type"] == "object"
    assert schema_b["type"] == "object"

    # But different descriptions
    assert schema_a["description"] == "Description A"
    assert schema_b["description"] == "Description B"

    # Modifying one shouldn't affect the other
    schema_a["properties"]["new_field"] = {"type": "integer"}
    assert "new_field" not in schema_b.get("properties", {})
