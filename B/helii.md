### Main Prompt (Turn 1)
The current OpenAPI to MCP conversion resolves schema references, but when schemas contain deeply nested references or circular references (which some OpenAPI generators produce), the resolution might not work correctly or could be inefficient. Improve the reference resolution logic to handle edge cases better and add detection for problematic reference patterns that could cause issues.

### Follow-up 1 (Turn 2)
The reference resolution improvements help, but I noticed that when resolving references, you're creating new schema dictionaries which can lead to memory issues with large OpenAPI specs. Please optimize the resolution to reuse resolved schemas where possible and add caching to avoid re-resolving the same references multiple times. Also, add logging when circular references are detected so users know about potential issues in their OpenAPI specs.

### Follow-up 2 (Turn 3)
The caching helps with performance, but I see that the resolved schema structure might still contain `$ref` strings in some edge cases. Please add validation after resolution to ensure all `$ref` references have been properly resolved, and raise a clear error message if any unresolved references remain. Also, make sure that the resolution preserves important schema metadata like `title`, `description`, and `examples` that might be defined at the reference level.

### Follow-up 3 (Turn 4)
The reference resolution is working well now, but I want to ensure it handles all the edge cases properly. Please add tests for complex scenarios including: deeply nested references (5+ levels), references within array items, references within `allOf`/`oneOf`/`anyOf` constructs, and schemas with both local and external references. Also, verify that the resolved schemas produce correct MCP tool definitions and that no information is lost during the resolution process.