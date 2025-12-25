import pytest
import base64
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI

from fastapi_mcp import FastApiMCP
from mcp.types import TextContent, ImageContent


@pytest.mark.asyncio
async def test_execute_api_tool_success(simple_fastapi_app: FastAPI):
    """Test successful execution of an API tool."""
    mcp = FastApiMCP(simple_fastapi_app)
    
    # Mock the HTTP client response
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 1, "name": "Test Item"}
    mock_response.status_code = 200
    mock_response.text = '{"id": 1, "name": "Test Item"}'
    
    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}
    
    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )
    
    # Verify the result
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == '{\n  "id": 1,\n  "name": "Test Item"\n}'
    
    # Verify the HTTP client was called correctly
    mock_client.get.assert_called_once_with(
        "/items/1",
        params={},
        headers={}
    )


@pytest.mark.asyncio
async def test_execute_api_tool_with_query_params(simple_fastapi_app: FastAPI):
    """Test execution of an API tool with query parameters."""
    mcp = FastApiMCP(simple_fastapi_app)
    
    # Mock the HTTP client response
    mock_response = MagicMock()
    mock_response.json.return_value = [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]
    mock_response.status_code = 200
    mock_response.text = '[{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]'
    
    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    # Test parameters
    tool_name = "list_items"
    arguments = {"skip": 0, "limit": 2}
    
    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )
    
    # Verify the result
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    
    # Verify the HTTP client was called with query parameters
    mock_client.get.assert_called_once_with(
        "/items/",
        params={"skip": 0, "limit": 2},
        headers={}
    )


@pytest.mark.asyncio
async def test_execute_api_tool_with_body(simple_fastapi_app: FastAPI):
    """Test execution of an API tool with request body."""
    mcp = FastApiMCP(simple_fastapi_app)
    
    # Mock the HTTP client response
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 1, "name": "New Item"}
    mock_response.status_code = 200
    mock_response.text = '{"id": 1, "name": "New Item"}'
    
    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    
    # Test parameters
    tool_name = "create_item"
    arguments = {
        "item": {
            "id": 1,
            "name": "New Item",
            "price": 10.0,
            "tags": ["tag1"],
            "description": "New item description"
        }
    }
    
    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )
    
    # Verify the result
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    
    # Verify the HTTP client was called with the request body
    mock_client.post.assert_called_once_with(
        "/items/",
        params={},
        headers={},
        json=arguments
    )


@pytest.mark.asyncio
async def test_execute_api_tool_with_non_ascii_chars(simple_fastapi_app: FastAPI):
    """Test execution of an API tool with non-ASCII characters."""
    mcp = FastApiMCP(simple_fastapi_app)
    
    # Test data with both ASCII and non-ASCII characters
    test_data = {
        "id": 1,
        "name": "你好 World",  # Chinese characters + ASCII
        "price": 10.0,
        "tags": ["tag1", "标签2"],  # Chinese characters in tags
        "description": "这是一个测试描述"  # All Chinese characters
    }
    
    # Mock the HTTP client response
    mock_response = MagicMock()
    mock_response.json.return_value = test_data
    mock_response.status_code = 200
    mock_response.text = '{"id": 1, "name": "你好 World", "price": 10.0, "tags": ["tag1", "标签2"], "description": "这是一个测试描述"}'
    
    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}
    
    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )
    
    # Verify the result
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    
    # Verify that the response contains both ASCII and non-ASCII characters
    response_text = result[0].text
    assert "你好" in response_text  # Chinese characters preserved
    assert "World" in response_text  # ASCII characters preserved
    assert "标签2" in response_text  # Chinese characters in tags preserved
    assert "这是一个测试描述" in response_text  # All Chinese description preserved

    # Verify the HTTP client was called correctly
    mock_client.get.assert_called_once_with(
        "/items/1",
        params={},
        headers={}
    )


# Tests for content type handling


@pytest.mark.asyncio
async def test_execute_api_tool_plain_text_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns plain text."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Mock the HTTP client response with plain text
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/plain; charset=utf-8"}
    mock_response.text = "Hello, this is plain text response"
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "Hello, this is plain text response"


@pytest.mark.asyncio
async def test_execute_api_tool_html_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns HTML."""
    mcp = FastApiMCP(simple_fastapi_app)

    html_content = "<html><body><h1>Hello World</h1></body></html>"

    # Mock the HTTP client response with HTML
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/html; charset=utf-8"}
    mock_response.text = html_content
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert "[HTML Content]" in result[0].text
    assert html_content in result[0].text


@pytest.mark.asyncio
async def test_execute_api_tool_xml_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns XML."""
    mcp = FastApiMCP(simple_fastapi_app)

    xml_content = '<?xml version="1.0"?><root><item id="1"><name>Test</name></item></root>'

    # Mock the HTTP client response with XML
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/xml"}
    mock_response.content = xml_content.encode("utf-8")
    mock_response.text = xml_content
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    # The XML should be formatted (pretty-printed)
    assert "<root>" in result[0].text
    assert "<item" in result[0].text
    assert "<name>" in result[0].text


@pytest.mark.asyncio
async def test_execute_api_tool_image_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns an image."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Create a simple binary image content (fake PNG header)
    image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'

    # Mock the HTTP client response with an image
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "image/png"}
    mock_response.content = image_bytes
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is an ImageContent
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)
    assert result[0].mimeType == "image/png"
    # Verify the data is base64 encoded
    decoded_data = base64.standard_b64decode(result[0].data)
    assert decoded_data == image_bytes


@pytest.mark.asyncio
async def test_execute_api_tool_jpeg_image_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns a JPEG image."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Create a simple binary image content (fake JPEG header)
    image_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01'

    # Mock the HTTP client response with an image
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.content = image_bytes
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is an ImageContent
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)
    assert result[0].mimeType == "image/jpeg"


@pytest.mark.asyncio
async def test_execute_api_tool_binary_pdf_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns a PDF (binary content)."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Create fake PDF content
    pdf_bytes = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<\n>>\nendobj'

    # Mock the HTTP client response with a PDF
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/pdf"}
    mock_response.content = pdf_bytes
    mock_response.text = pdf_bytes.decode('latin-1')  # Will be decoded but not printable
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result contains binary content description
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    # PDFs may be detected as text due to the %PDF header being printable
    # The implementation tries to decode as text first


@pytest.mark.asyncio
async def test_execute_api_tool_json_ld_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns JSON-LD."""
    mcp = FastApiMCP(simple_fastapi_app)

    json_ld_data = {
        "@context": "https://schema.org",
        "@type": "Person",
        "name": "John Doe"
    }

    # Mock the HTTP client response with JSON-LD
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/ld+json"}
    mock_response.json.return_value = json_ld_data
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is formatted JSON
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert "@context" in result[0].text
    assert "schema.org" in result[0].text


@pytest.mark.asyncio
async def test_execute_api_tool_custom_json_media_type(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns a custom JSON media type (e.g., application/vnd.api+json)."""
    mcp = FastApiMCP(simple_fastapi_app)

    api_data = {"data": {"type": "articles", "id": "1"}}

    # Mock the HTTP client response with custom JSON type
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/vnd.api+json"}
    mock_response.json.return_value = api_data
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is formatted JSON
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert '"data"' in result[0].text
    assert '"articles"' in result[0].text


@pytest.mark.asyncio
async def test_execute_api_tool_csv_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns CSV."""
    mcp = FastApiMCP(simple_fastapi_app)

    csv_content = "id,name,price\n1,Item 1,10.00\n2,Item 2,20.00"

    # Mock the HTTP client response with CSV
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/csv"}
    mock_response.text = csv_content
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is returned as text
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == csv_content


@pytest.mark.asyncio
async def test_execute_api_tool_no_content_type_header(simple_fastapi_app: FastAPI):
    """Test execution when response has no Content-Type header."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Mock the HTTP client response without content-type
    mock_response = MagicMock()
    mock_response.headers = {}  # No content-type
    mock_response.content = b'Some binary data'
    mock_response.text = 'Some binary data'
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Should handle gracefully - will try to decode as text
    assert len(result) == 1
    assert isinstance(result[0], TextContent)


@pytest.mark.asyncio
async def test_execute_api_tool_xml_with_custom_media_type(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns XML with custom media type (e.g., application/atom+xml)."""
    mcp = FastApiMCP(simple_fastapi_app)

    atom_content = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Test Feed</title></feed>'

    # Mock the HTTP client response with Atom XML
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/atom+xml"}
    mock_response.content = atom_content.encode("utf-8")
    mock_response.text = atom_content
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is formatted XML
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert "<feed" in result[0].text
    assert "<title>" in result[0].text


@pytest.mark.asyncio
async def test_parse_content_type_with_charset(simple_fastapi_app: FastAPI):
    """Test parsing content type header with charset."""
    mcp = FastApiMCP(simple_fastapi_app)

    media_type, charset = mcp._parse_content_type("application/json; charset=utf-8")
    assert media_type == "application/json"
    assert charset == "utf-8"


@pytest.mark.asyncio
async def test_parse_content_type_without_charset(simple_fastapi_app: FastAPI):
    """Test parsing content type header without charset."""
    mcp = FastApiMCP(simple_fastapi_app)

    media_type, charset = mcp._parse_content_type("image/png")
    assert media_type == "image/png"
    assert charset is None


@pytest.mark.asyncio
async def test_parse_content_type_none(simple_fastapi_app: FastAPI):
    """Test parsing None content type header."""
    mcp = FastApiMCP(simple_fastapi_app)

    media_type, charset = mcp._parse_content_type(None)
    assert media_type == "application/octet-stream"
    assert charset is None


@pytest.mark.asyncio
async def test_parse_content_type_with_multiple_params(simple_fastapi_app: FastAPI):
    """Test parsing content type header with multiple parameters."""
    mcp = FastApiMCP(simple_fastapi_app)

    media_type, charset = mcp._parse_content_type("text/html; charset=UTF-8; boundary=something")
    assert media_type == "text/html"
    assert charset == "UTF-8"


@pytest.mark.asyncio
async def test_execute_api_tool_empty_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns an empty response (e.g., 204 No Content)."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Mock the HTTP client response with empty content (204 No Content)
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/json"}
    mock_response.content = b""  # Empty content
    mock_response.status_code = 204

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.delete.return_value = mock_response

    # Test parameters - simulate a delete operation
    tool_name = "delete_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is an empty text response
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == ""


@pytest.mark.asyncio
async def test_execute_api_tool_svg_image_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns an SVG image."""
    mcp = FastApiMCP(simple_fastapi_app)

    # SVG is an image type but text-based
    svg_content = b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40"/></svg>'

    # Mock the HTTP client response with SVG
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "image/svg+xml"}
    mock_response.content = svg_content
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is an ImageContent (SVG is image/*)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)
    assert result[0].mimeType == "image/svg+xml"


@pytest.mark.asyncio
async def test_execute_api_tool_json_with_invalid_json_content(simple_fastapi_app: FastAPI):
    """Test execution when content-type is JSON but content is not valid JSON."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Mock the HTTP client response with invalid JSON
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_response.content = b"This is not JSON"
    mock_response.text = "This is not JSON"
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Should fall through to text handling
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "This is not JSON"


@pytest.mark.asyncio
async def test_execute_api_tool_large_image_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns a large image (exceeds size limit)."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Create a large fake image (larger than _MAX_IMAGE_SIZE)
    large_image_bytes = b'\x89PNG' + (b'\x00' * (6 * 1024 * 1024))  # ~6MB

    # Mock the HTTP client response with a large image
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "image/png"}
    mock_response.content = large_image_bytes
    mock_response.status_code = 200

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is a TextContent with summary (not ImageContent)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert "[Image content: image/png" in result[0].text
    assert "too large" in result[0].text


@pytest.mark.asyncio
async def test_execute_api_tool_large_binary_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns large binary content (exceeds size limit)."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Create large binary content (larger than _MAX_BINARY_SIZE_FOR_BASE64)
    large_binary = b'\x00\x01\x02' * (500 * 1024)  # ~1.5MB

    # Mock the HTTP client response with large binary
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/octet-stream"}
    mock_response.content = large_binary
    mock_response.text = None  # Will raise exception when accessed
    mock_response.status_code = 200

    # Make text access raise an exception (simulating true binary)
    type(mock_response).text = property(lambda self: (_ for _ in ()).throw(UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')))

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result is a summary (no base64 data)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert "[Binary content: application/octet-stream" in result[0].text
    assert "too large" in result[0].text
    assert "Base64" not in result[0].text


@pytest.mark.asyncio
async def test_execute_api_tool_small_binary_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns small binary content (within size limit)."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Create small binary content
    small_binary = b'\x00\x01\x02\x03\x04\x05'

    # Mock the HTTP client response with small binary
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/octet-stream"}
    mock_response.content = small_binary
    mock_response.status_code = 200

    # Make text access raise an exception (simulating true binary)
    type(mock_response).text = property(lambda self: (_ for _ in ()).throw(UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')))

    # Mock the HTTP client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Test parameters
    tool_name = "get_item"
    arguments = {"item_id": 1}

    # Execute the tool
    with patch.object(mcp, '_http_client', mock_client):
        result = await mcp._execute_api_tool(
            client=mock_client,
            tool_name=tool_name,
            arguments=arguments,
            operation_map=mcp.operation_map
        )

    # Verify the result includes base64 data
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert "[Binary content: application/octet-stream" in result[0].text
    assert "Base64 encoded data:" in result[0].text
    # Verify the base64 decodes back to original
    assert base64.standard_b64encode(small_binary).decode("ascii") in result[0].text
