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


def test_deeply_nested_references_7_levels():
    """Test handling of very deeply nested references (7 levels)."""
    schema = {
        "components": {
            "schemas": {
                "Level1": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/Level2"}, "value": {"type": "string"}},
                },
                "Level2": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/Level3"}, "value": {"type": "string"}},
                },
                "Level3": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/Level4"}, "value": {"type": "string"}},
                },
                "Level4": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/Level5"}, "value": {"type": "string"}},
                },
                "Level5": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/Level6"}, "value": {"type": "string"}},
                },
                "Level6": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/Level7"}, "value": {"type": "string"}},
                },
                "Level7": {
                    "type": "object",
                    "properties": {"final": {"type": "boolean"}},
                },
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

    # Should resolve without issues
    assert result.is_fully_resolved
    assert len(result.circular_refs) == 0

    # Verify the chain is fully resolved
    level1 = result.schema["paths"]["/deep"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert level1.get("type") == "object"
    assert "next" in level1["properties"]

    # Traverse down the chain
    current = level1
    for i in range(6):  # 6 more levels
        assert "next" in current["properties"], f"Missing 'next' at level {i+1}"
        current = current["properties"]["next"]
        assert current.get("type") == "object", f"Type not resolved at level {i+2}"

    # Final level should have 'final' property
    assert "final" in current["properties"]
    assert current["properties"]["final"].get("type") == "boolean"


def test_references_in_nested_array_items():
    """Test references deeply nested within array items."""
    schema = {
        "components": {
            "schemas": {
                "Tag": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "color": {"type": "string"}},
                },
                "Item": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "tags": {"type": "array", "items": {"$ref": "#/components/schemas/Tag"}},
                    },
                },
                "Container": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"$ref": "#/components/schemas/Item"}},
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "tags": {"type": "array", "items": {"$ref": "#/components/schemas/Tag"}}
                            },
                        },
                    },
                },
            }
        },
        "paths": {
            "/containers": {
                "get": {
                    "operationId": "list_containers",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Container"},
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

    assert result.is_fully_resolved
    assert len(result.circular_refs) == 0

    # Verify nested array items are resolved
    response_schema = result.schema["paths"]["/containers"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    assert response_schema.get("type") == "array"

    container = response_schema["items"]
    assert container.get("type") == "object"

    # Check items array
    items_array = container["properties"]["items"]
    assert items_array.get("type") == "array"
    item = items_array["items"]
    assert item.get("type") == "object"

    # Check tags inside item
    tags_in_item = item["properties"]["tags"]
    assert tags_in_item.get("type") == "array"
    tag = tags_in_item["items"]
    assert tag.get("type") == "object"
    assert "name" in tag["properties"]
    assert "color" in tag["properties"]


def test_allof_with_multiple_refs():
    """Test allOf construct with multiple references."""
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
                    "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                },
                "Resource": {
                    "allOf": [
                        {"$ref": "#/components/schemas/Identifiable"},
                        {"$ref": "#/components/schemas/Timestamped"},
                        {"$ref": "#/components/schemas/Named"},
                        {"type": "object", "properties": {"status": {"type": "string", "enum": ["active", "inactive"]}}},
                    ]
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    resource = result["components"]["schemas"]["Resource"]
    assert "allOf" in resource
    assert len(resource["allOf"]) == 4

    # Check each allOf item is resolved
    assert resource["allOf"][0].get("type") == "object"
    assert "id" in resource["allOf"][0]["properties"]

    assert resource["allOf"][1].get("type") == "object"
    assert "created_at" in resource["allOf"][1]["properties"]

    assert resource["allOf"][2].get("type") == "object"
    assert "name" in resource["allOf"][2]["properties"]

    assert resource["allOf"][3].get("type") == "object"
    assert "status" in resource["allOf"][3]["properties"]


def test_oneof_with_refs():
    """Test oneOf construct with references."""
    schema = {
        "components": {
            "schemas": {
                "Cat": {
                    "type": "object",
                    "properties": {"meow_volume": {"type": "integer"}, "name": {"type": "string"}},
                },
                "Dog": {
                    "type": "object",
                    "properties": {"bark_volume": {"type": "integer"}, "name": {"type": "string"}},
                },
                "Pet": {"oneOf": [{"$ref": "#/components/schemas/Cat"}, {"$ref": "#/components/schemas/Dog"}]},
            }
        },
        "paths": {
            "/pets": {
                "post": {
                    "operationId": "create_pet",
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}}
                    },
                    "responses": {"201": {"description": "Created"}},
                }
            }
        },
    }

    result = resolve_schema_references(schema, schema)

    pet = result["components"]["schemas"]["Pet"]
    assert "oneOf" in pet
    assert len(pet["oneOf"]) == 2

    # Check Cat is resolved
    assert pet["oneOf"][0].get("type") == "object"
    assert "meow_volume" in pet["oneOf"][0]["properties"]

    # Check Dog is resolved
    assert pet["oneOf"][1].get("type") == "object"
    assert "bark_volume" in pet["oneOf"][1]["properties"]


def test_anyof_with_refs():
    """Test anyOf construct with references."""
    schema = {
        "components": {
            "schemas": {
                "StringId": {"type": "object", "properties": {"id": {"type": "string"}}},
                "IntegerId": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "FlexibleId": {
                    "anyOf": [{"$ref": "#/components/schemas/StringId"}, {"$ref": "#/components/schemas/IntegerId"}]
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    flexible = result["components"]["schemas"]["FlexibleId"]
    assert "anyOf" in flexible
    assert len(flexible["anyOf"]) == 2

    # Both should be resolved
    assert flexible["anyOf"][0].get("type") == "object"
    assert flexible["anyOf"][0]["properties"]["id"].get("type") == "string"

    assert flexible["anyOf"][1].get("type") == "object"
    assert flexible["anyOf"][1]["properties"]["id"].get("type") == "integer"


def test_nested_composition_refs():
    """Test deeply nested composition (allOf within oneOf within allOf)."""
    schema = {
        "components": {
            "schemas": {
                "Base": {"type": "object", "properties": {"base_prop": {"type": "string"}}},
                "ExtA": {"type": "object", "properties": {"ext_a": {"type": "string"}}},
                "ExtB": {"type": "object", "properties": {"ext_b": {"type": "string"}}},
                "Variant1": {"allOf": [{"$ref": "#/components/schemas/Base"}, {"$ref": "#/components/schemas/ExtA"}]},
                "Variant2": {"allOf": [{"$ref": "#/components/schemas/Base"}, {"$ref": "#/components/schemas/ExtB"}]},
                "Complex": {
                    "allOf": [
                        {
                            "oneOf": [
                                {"$ref": "#/components/schemas/Variant1"},
                                {"$ref": "#/components/schemas/Variant2"},
                            ]
                        },
                        {"type": "object", "properties": {"extra": {"type": "boolean"}}},
                    ]
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    complex_schema = result["components"]["schemas"]["Complex"]
    assert "allOf" in complex_schema
    assert len(complex_schema["allOf"]) == 2

    # Check oneOf within allOf
    one_of_part = complex_schema["allOf"][0]
    assert "oneOf" in one_of_part
    assert len(one_of_part["oneOf"]) == 2

    # Check Variant1 is resolved (allOf)
    variant1 = one_of_part["oneOf"][0]
    assert "allOf" in variant1
    assert variant1["allOf"][0].get("type") == "object"
    assert "base_prop" in variant1["allOf"][0]["properties"]


def test_mixed_local_and_external_refs():
    """Test schema with both local and external references."""
    schema = {
        "components": {
            "schemas": {
                "LocalModel": {"type": "object", "properties": {"local": {"type": "string"}}},
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
                                            "local_data": {"$ref": "#/components/schemas/LocalModel"},
                                            "external_data": {"$ref": "https://example.com/schemas/External.json"},
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

    # Local ref should be resolved
    response = result.schema["paths"]["/mixed"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    local_data = response["properties"]["local_data"]
    assert local_data.get("type") == "object"
    assert "local" in local_data["properties"]

    # External ref should remain unresolved
    external_data = response["properties"]["external_data"]
    assert "$ref" in external_data
    assert external_data["$ref"] == "https://example.com/schemas/External.json"

    # Should have unresolved refs
    assert result.has_unresolved_refs
    assert "https://example.com/schemas/External.json" in result.unresolved_refs


def test_parameter_refs():
    """Test references to parameters component."""
    schema = {
        "components": {
            "parameters": {
                "PageParam": {
                    "name": "page",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1},
                    "description": "Page number",
                },
                "SizeParam": {
                    "name": "size",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 100},
                    "description": "Page size",
                },
            }
        },
        "paths": {
            "/items": {
                "get": {
                    "operationId": "list_items",
                    "parameters": [
                        {"$ref": "#/components/parameters/PageParam"},
                        {"$ref": "#/components/parameters/SizeParam"},
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    }

    result = resolve_schema_references(schema, schema)

    params = result["paths"]["/items"]["get"]["parameters"]
    assert len(params) == 2

    # Check params are resolved
    assert params[0].get("name") == "page"
    assert params[0].get("in") == "query"
    assert params[0]["schema"].get("type") == "integer"

    assert params[1].get("name") == "size"
    assert params[1].get("in") == "query"


def test_mcp_tool_generation_with_complex_refs():
    """Test that MCP tools are correctly generated from schemas with complex references."""
    schema = {
        "openapi": "3.1.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "components": {
            "schemas": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                    },
                    "required": ["street", "city"],
                },
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Person's full name"},
                        "email": {"type": "string", "format": "email"},
                        "address": {"$ref": "#/components/schemas/Address"},
                    },
                    "required": ["name", "email"],
                },
            }
        },
        "paths": {
            "/persons": {
                "post": {
                    "operationId": "create_person",
                    "summary": "Create a new person",
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

    # Person properties should be in the input schema
    props = input_schema["properties"]
    assert "name" in props
    assert "email" in props
    assert "address" in props

    # Address should be resolved
    address_prop = props["address"]
    assert address_prop.get("type") == "object"
    assert "properties" in address_prop
    assert "street" in address_prop["properties"]
    assert "city" in address_prop["properties"]


def test_mcp_tool_with_array_of_refs():
    """Test MCP tool generation with array items containing references."""
    schema = {
        "openapi": "3.1.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "components": {
            "schemas": {
                "OrderItem": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "quantity": {"type": "integer", "minimum": 1},
                        "price": {"type": "number"},
                    },
                    "required": ["product_id", "quantity"],
                }
            }
        },
        "paths": {
            "/orders": {
                "post": {
                    "operationId": "create_order",
                    "summary": "Create order",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "customer_id": {"type": "string"},
                                        "items": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/OrderItem"},
                                        },
                                    },
                                    "required": ["customer_id", "items"],
                                }
                            }
                        },
                    },
                    "responses": {"201": {"description": "Created"}},
                }
            }
        },
    }

    tools, operation_map = convert_openapi_to_mcp_tools(schema)

    assert len(tools) == 1
    tool = tools[0]

    props = tool.inputSchema["properties"]
    assert "customer_id" in props
    assert "items" in props

    # Check items array is properly resolved
    items_prop = props["items"]
    assert items_prop.get("type") == "array"
    assert "items" in items_prop

    item_schema = items_prop["items"]
    assert item_schema.get("type") == "object"
    assert "product_id" in item_schema["properties"]
    assert "quantity" in item_schema["properties"]


def test_info_preservation_during_resolution():
    """Test that schema information is not lost during resolution."""
    schema = {
        "components": {
            "schemas": {
                "DetailedModel": {
                    "type": "object",
                    "title": "Detailed Model",
                    "description": "A model with lots of details",
                    "properties": {
                        "id": {"type": "integer", "description": "Unique identifier", "minimum": 1},
                        "name": {
                            "type": "string",
                            "description": "Display name",
                            "minLength": 1,
                            "maxLength": 100,
                            "pattern": "^[a-zA-Z]+$",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "uniqueItems": True,
                            "minItems": 1,
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["id", "name"],
                    "additionalProperties": False,
                    "example": {"id": 1, "name": "Example", "tags": ["tag1"]},
                }
            }
        },
        "paths": {
            "/models": {
                "get": {
                    "operationId": "get_model",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/DetailedModel"}}
                            }
                        }
                    },
                }
            }
        },
    }

    result = resolve_schema_references(schema, schema)

    model = result["paths"]["/models"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]

    # Check all details are preserved
    assert model.get("type") == "object"
    assert model.get("title") == "Detailed Model"
    assert model.get("description") == "A model with lots of details"
    assert model.get("required") == ["id", "name"]
    assert model.get("additionalProperties") is False
    assert model.get("example") == {"id": 1, "name": "Example", "tags": ["tag1"]}

    # Check property details
    id_prop = model["properties"]["id"]
    assert id_prop.get("description") == "Unique identifier"
    assert id_prop.get("minimum") == 1

    name_prop = model["properties"]["name"]
    assert name_prop.get("minLength") == 1
    assert name_prop.get("maxLength") == 100
    assert name_prop.get("pattern") == "^[a-zA-Z]+$"

    tags_prop = model["properties"]["tags"]
    assert tags_prop.get("uniqueItems") is True
    assert tags_prop.get("minItems") == 1


def test_discriminator_preservation():
    """Test that discriminator information is preserved."""
    schema = {
        "components": {
            "schemas": {
                "Pet": {
                    "type": "object",
                    "discriminator": {"propertyName": "petType", "mapping": {"cat": "#/components/schemas/Cat"}},
                    "properties": {"petType": {"type": "string"}},
                    "required": ["petType"],
                },
                "Cat": {
                    "allOf": [
                        {"$ref": "#/components/schemas/Pet"},
                        {"type": "object", "properties": {"meow": {"type": "boolean"}}},
                    ]
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    pet = result["components"]["schemas"]["Pet"]
    assert "discriminator" in pet
    assert pet["discriminator"]["propertyName"] == "petType"


def test_multiple_circular_chains():
    """Test handling of multiple independent circular reference chains."""
    schema = {
        "components": {
            "schemas": {
                # Chain 1: A -> B -> A
                "ChainA1": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/ChainA2"}},
                },
                "ChainA2": {
                    "type": "object",
                    "properties": {"back": {"$ref": "#/components/schemas/ChainA1"}},
                },
                # Chain 2: X -> Y -> Z -> X
                "ChainB1": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/ChainB2"}},
                },
                "ChainB2": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/ChainB3"}},
                },
                "ChainB3": {
                    "type": "object",
                    "properties": {"back": {"$ref": "#/components/schemas/ChainB1"}},
                },
            }
        }
    }

    result = resolve_schema_references_with_details(schema, schema)

    # Should detect multiple circular refs
    assert len(result.circular_refs) >= 2

    # Analysis should also detect them
    analysis = analyze_schema_references(schema)
    assert len(analysis.circular_refs) >= 2


def test_ref_inside_additional_properties():
    """Test references inside additionalProperties."""
    schema = {
        "components": {
            "schemas": {
                "Value": {"type": "object", "properties": {"data": {"type": "string"}}},
                "Map": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/components/schemas/Value"},
                },
            }
        }
    }

    result = resolve_schema_references(schema, schema)

    map_schema = result["components"]["schemas"]["Map"]
    assert map_schema.get("type") == "object"
    assert "additionalProperties" in map_schema

    additional = map_schema["additionalProperties"]
    assert additional.get("type") == "object"
    assert "data" in additional["properties"]
