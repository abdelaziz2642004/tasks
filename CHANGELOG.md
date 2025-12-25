## Prompt Set 6: Response Content Type Handling

### Main Prompt (Turn 1)
Currently, the MCP tool execution assumes API responses are JSON and tries to parse them as such. However, some FastAPI endpoints might return other content types like plain text, XML, or binary data. The current implementation might fail or return poorly formatted results for non-JSON responses. Add support for handling different response content types appropriately, formatting them in a way that's useful for LLM consumption.

### Follow-up 1 (Turn 2)
The content type handling you added works for text-based responses, but for binary content like images or PDFs, returning the raw bytes as text isn't useful. Please add logic to detect binary content types and either: (1) return a summary message indicating the content type and size for large binary responses, or (2) encode small binary responses in base64 with appropriate metadata. Also, ensure that the Content-Type header from the API response is properly respected.

### Follow-up 2 (Turn 3)
I see that you're handling different content types, but the logic for determining what's "binary" vs "text" could be more robust. Some APIs might return JSON with an incorrect Content-Type header, or vice versa. Please improve the content type detection to check both the Content-Type header and the actual response content, and add fallback logic for cases where the Content-Type is missing or ambiguous. Also, make sure XML responses are formatted in a readable way for LLMs.

### Follow-up 3 (Turn 4)
The content type handling is much better now, but I noticed that when an API returns an error response with a different content type (like HTML error pages), your new content type handling might interfere with the error extraction logic we have. Please ensure that error responses are still properly handled regardless of their content type, and that the error message extraction works correctly for HTML, XML, and other non-JSON error responses. Add tests covering various content types in both success and error scenarios.