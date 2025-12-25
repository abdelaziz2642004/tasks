import pytest
import base64
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI

from fastapi_mcp import FastApiMCP
from fastapi_mcp.server import _format_response_for_llm
from mcp.types import TextContent, ImageContent


@pytest.mark.asyncio
async def test_execute_api_tool_success(simple_fastapi_app: FastAPI):
    """Test successful execution of an API tool."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Mock the HTTP client response
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/json"}
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
    mock_response.headers = {"content-type": "application/json"}
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
    mock_response.headers = {"content-type": "application/json"}
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
    mock_response.headers = {"content-type": "application/json"}
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


# Tests for _format_response_for_llm helper function

class TestFormatResponseForLLM:
    """Tests for the _format_response_for_llm function that handles different content types."""

    def test_json_response(self):
        """Test that JSON responses are pretty-printed."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"key": "value", "number": 42}

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == '{\n  "key": "value",\n  "number": 42\n}'

    def test_json_response_with_charset(self):
        """Test that JSON responses with charset are handled correctly."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json; charset=utf-8"}
        mock_response.json.return_value = {"message": "hello"}

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert '"message": "hello"' in result[0].text

    def test_hal_json_response(self):
        """Test that HAL+JSON responses are handled as JSON."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/hal+json"}
        mock_response.json.return_value = {"_links": {"self": {"href": "/api"}}}

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "_links" in result[0].text

    def test_custom_json_media_type(self):
        """Test that custom +json media types are handled as JSON."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/vnd.myapp.v1+json"}
        mock_response.json.return_value = {"data": "test"}

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert '"data": "test"' in result[0].text

    def test_plain_text_response(self):
        """Test that plain text responses are returned as-is."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "Hello, World!"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Hello, World!"

    def test_html_response(self):
        """Test that HTML responses are returned as text."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html><body><h1>Hello</h1></body></html>"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "<h1>Hello</h1>" in result[0].text

    def test_xhtml_response(self):
        """Test that XHTML responses are returned as text."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/xhtml+xml"}
        mock_response.text = '<?xml version="1.0"?><html><body>Test</body></html>'

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "<body>Test</body>" in result[0].text

    def test_xml_response(self):
        """Test that XML responses are returned as text."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/xml"}
        mock_response.text = '<?xml version="1.0"?><root><item>Test</item></root>'

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "<item>Test</item>" in result[0].text

    def test_text_xml_response(self):
        """Test that text/xml responses are returned as text."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/xml"}
        mock_response.text = "<data><value>123</value></data>"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "<value>123</value>" in result[0].text

    def test_custom_xml_media_type(self):
        """Test that custom +xml media types are handled as XML."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/atom+xml"}
        mock_response.text = '<?xml version="1.0"?><feed><entry>Test</entry></feed>'

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "<entry>Test</entry>" in result[0].text

    def test_image_png_response(self):
        """Test that PNG images are returned as ImageContent."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], ImageContent)
        assert result[0].mimeType == "image/png"
        assert result[0].data == base64.standard_b64encode(b"\x89PNG\r\n\x1a\n").decode("utf-8")

    def test_image_jpeg_response(self):
        """Test that JPEG images are returned as ImageContent."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"\xff\xd8\xff\xe0"  # JPEG magic bytes

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], ImageContent)
        assert result[0].mimeType == "image/jpeg"

    def test_image_gif_response(self):
        """Test that GIF images are returned as ImageContent."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/gif"}
        mock_response.content = b"GIF89a"  # GIF magic bytes

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], ImageContent)
        assert result[0].mimeType == "image/gif"

    def test_image_webp_response(self):
        """Test that WebP images are returned as ImageContent."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/webp"}
        mock_response.content = b"RIFF\x00\x00\x00\x00WEBP"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], ImageContent)
        assert result[0].mimeType == "image/webp"

    def test_image_svg_response(self):
        """Test that SVG images are returned as ImageContent."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/svg+xml"}
        mock_response.content = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], ImageContent)
        assert result[0].mimeType == "image/svg+xml"

    def test_binary_octet_stream_response(self):
        """Test that binary octet-stream responses are base64-encoded with metadata."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.content = b"\x00\x01\x02\x03\x04\x05"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Binary response" in result[0].text
        assert "application/octet-stream" in result[0].text
        assert "6 bytes" in result[0].text
        assert base64.standard_b64encode(b"\x00\x01\x02\x03\x04\x05").decode("utf-8") in result[0].text

    def test_pdf_response(self):
        """Test that PDF responses are base64-encoded with metadata."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.content = b"%PDF-1.4 test content"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Binary response" in result[0].text
        assert "application/pdf" in result[0].text

    def test_text_csv_response(self):
        """Test that CSV text responses are returned as text."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/csv"}
        mock_response.text = "name,value\ntest,123"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "name,value\ntest,123"

    def test_json_decode_error_fallback(self):
        """Test that invalid JSON falls through to text handling."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.side_effect = ValueError("Invalid JSON")  # json.JSONDecodeError inherits from ValueError
        mock_response.content = b"not valid json"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        # Should fall through to binary handling since it's not text/*
        assert "Binary response" in result[0].text

    def test_empty_response(self):
        """Test that empty responses return empty string."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": ""}
        mock_response.content = b""
        mock_response.text = ""

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        # Empty content returns empty string (e.g., for 204 No Content responses)
        assert result[0].text == ""

    def test_no_content_type_header(self):
        """Test handling when content-type header is missing."""
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.content = b"some data"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Binary response" in result[0].text
        assert "unknown" in result[0].text

    def test_unicode_in_text_response(self):
        """Test that unicode characters are preserved in text responses."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "Hello, 世界! 🌍"

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "世界" in result[0].text
        assert "🌍" in result[0].text

    def test_unicode_in_json_response(self):
        """Test that unicode characters are preserved in JSON responses."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"greeting": "你好", "emoji": "🎉"}

        result = _format_response_for_llm(mock_response)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "你好" in result[0].text
        assert "🎉" in result[0].text


@pytest.mark.asyncio
async def test_execute_api_tool_with_plain_text_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns plain text."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Mock the HTTP client response
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/plain"}
    mock_response.text = "Operation completed successfully"
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
    assert result[0].text == "Operation completed successfully"


@pytest.mark.asyncio
async def test_execute_api_tool_with_xml_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns XML."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Mock the HTTP client response
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/xml"}
    mock_response.text = '<?xml version="1.0"?><item><id>1</id><name>Test</name></item>'
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
    assert "<item>" in result[0].text
    assert "<name>Test</name>" in result[0].text


@pytest.mark.asyncio
async def test_execute_api_tool_with_image_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns an image."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Mock the HTTP client response with PNG image
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "image/png"}
    mock_response.content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
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
    assert isinstance(result[0], ImageContent)
    assert result[0].mimeType == "image/png"
    # Verify the data is base64 encoded
    expected_data = base64.standard_b64encode(mock_response.content).decode("utf-8")
    assert result[0].data == expected_data


@pytest.mark.asyncio
async def test_execute_api_tool_with_html_response(simple_fastapi_app: FastAPI):
    """Test execution of an API tool that returns HTML."""
    mcp = FastApiMCP(simple_fastapi_app)

    # Mock the HTTP client response
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/html; charset=utf-8"}
    mock_response.text = "<!DOCTYPE html><html><body><h1>Welcome</h1></body></html>"
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
    assert "<h1>Welcome</h1>" in result[0].text
