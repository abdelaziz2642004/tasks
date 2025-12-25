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
