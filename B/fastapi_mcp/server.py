import json
import base64
import httpx
from typing import Dict, Optional, Any, List, Union, Literal, Sequence
from typing_extensions import Annotated, Doc

from fastapi import FastAPI, Request, APIRouter, params
from fastapi.openapi.utils import get_openapi
from mcp.server.lowlevel.server import Server
import mcp.types as types

from fastapi_mcp.openapi.convert import convert_openapi_to_mcp_tools
from fastapi_mcp.transport.sse import FastApiSseTransport
from fastapi_mcp.transport.http import FastApiHttpSessionManager
from fastapi_mcp.types import HTTPRequestInfo, AuthConfig

import logging


logger = logging.getLogger(__name__)


class FastApiMCP:
    """
    Create an MCP server from a FastAPI app.
    """

    def __init__(
        self,
        fastapi: Annotated[
            FastAPI,
            Doc("The FastAPI application to create an MCP server from"),
        ],
        name: Annotated[
            Optional[str],
            Doc("Name for the MCP server (defaults to app.title)"),
        ] = None,
        description: Annotated[
            Optional[str],
            Doc("Description for the MCP server (defaults to app.description)"),
        ] = None,
        describe_all_responses: Annotated[
            bool,
            Doc("Whether to include all possible response schemas in tool descriptions"),
        ] = False,
        describe_full_response_schema: Annotated[
            bool,
            Doc("Whether to include full json schema for responses in tool descriptions"),
        ] = False,
        http_client: Annotated[
            Optional[httpx.AsyncClient],
            Doc(
                """
                Optional custom HTTP client to use for API calls to the FastAPI app.
                Has to be an instance of `httpx.AsyncClient`.
                """
            ),
        ] = None,
        include_operations: Annotated[
            Optional[List[str]],
            Doc("List of operation IDs to include as MCP tools. Cannot be used with exclude_operations."),
        ] = None,
        exclude_operations: Annotated[
            Optional[List[str]],
            Doc("List of operation IDs to exclude from MCP tools. Cannot be used with include_operations."),
        ] = None,
        include_tags: Annotated[
            Optional[List[str]],
            Doc("List of tags to include as MCP tools. Cannot be used with exclude_tags."),
        ] = None,
        exclude_tags: Annotated[
            Optional[List[str]],
            Doc("List of tags to exclude from MCP tools. Cannot be used with include_tags."),
        ] = None,
        auth_config: Annotated[
            Optional[AuthConfig],
            Doc("Configuration for MCP authentication"),
        ] = None,
        headers: Annotated[
            List[str],
            Doc(
                """
                List of HTTP header names to forward from the incoming MCP request into each tool invocation.
                Only headers in this allowlist will be forwarded. Defaults to ['authorization'].
                """
            ),
        ] = ["authorization"],
    ):
        # Validate operation and tag filtering options
        if include_operations is not None and exclude_operations is not None:
            raise ValueError("Cannot specify both include_operations and exclude_operations")

        if include_tags is not None and exclude_tags is not None:
            raise ValueError("Cannot specify both include_tags and exclude_tags")

        self.operation_map: Dict[str, Dict[str, Any]]
        self.tools: List[types.Tool]
        self.server: Server

        self.fastapi = fastapi
        self.name = name or self.fastapi.title or "FastAPI MCP"
        self.description = description or self.fastapi.description

        self._base_url = "http://apiserver"
        self._describe_all_responses = describe_all_responses
        self._describe_full_response_schema = describe_full_response_schema
        self._include_operations = include_operations
        self._exclude_operations = exclude_operations
        self._include_tags = include_tags
        self._exclude_tags = exclude_tags
        self._auth_config = auth_config

        if self._auth_config:
            self._auth_config = self._auth_config.model_validate(self._auth_config)

        self._http_client = http_client or httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.fastapi, raise_app_exceptions=False),
            base_url=self._base_url,
            timeout=10.0,
        )

        self._forward_headers = {h.lower() for h in headers}
        self._http_transport: FastApiHttpSessionManager | None = None  # Store reference to HTTP transport for cleanup

        self.setup_server()

    def setup_server(self) -> None:
        openapi_schema = get_openapi(
            title=self.fastapi.title,
            version=self.fastapi.version,
            openapi_version=self.fastapi.openapi_version,
            description=self.fastapi.description,
            routes=self.fastapi.routes,
        )

        all_tools, self.operation_map = convert_openapi_to_mcp_tools(
            openapi_schema,
            describe_all_responses=self._describe_all_responses,
            describe_full_response_schema=self._describe_full_response_schema,
        )

        # Filter tools based on operation IDs and tags
        self.tools = self._filter_tools(all_tools, openapi_schema)

        mcp_server: Server = Server(self.name, self.description)

        @mcp_server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return self.tools

        @mcp_server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> List[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
            # Extract HTTP request info from MCP context
            http_request_info = None
            try:
                # Access the MCP server's request context to get the original HTTP Request
                request_context = mcp_server.request_context

                if request_context and hasattr(request_context, "request"):
                    http_request = request_context.request

                    if http_request and hasattr(http_request, "method"):
                        http_request_info = HTTPRequestInfo(
                            method=http_request.method,
                            path=http_request.url.path,
                            headers=dict(http_request.headers),
                            cookies=http_request.cookies,
                            query_params=dict(http_request.query_params),
                            body=None,
                        )
                        logger.debug(
                            f"Extracted HTTP request info from context: {http_request_info.method} {http_request_info.path}"
                        )
            except (LookupError, AttributeError) as e:
                logger.error(f"Could not extract HTTP request info from context: {e}")

            return await self._execute_api_tool(
                client=self._http_client,
                tool_name=name,
                arguments=arguments,
                operation_map=self.operation_map,
                http_request_info=http_request_info,
            )

        self.server = mcp_server

    def _register_mcp_connection_endpoint_sse(
        self,
        router: FastAPI | APIRouter,
        transport: FastApiSseTransport,
        mount_path: str,
        dependencies: Optional[Sequence[params.Depends]],
    ):
        @router.get(mount_path, include_in_schema=False, operation_id="mcp_connection", dependencies=dependencies)
        async def handle_mcp_connection(request: Request):
            async with transport.connect_sse(request.scope, request.receive, request._send) as (reader, writer):
                await self.server.run(
                    reader,
                    writer,
                    self.server.create_initialization_options(notification_options=None, experimental_capabilities={}),
                    raise_exceptions=False,
                )

    def _register_mcp_messages_endpoint_sse(
        self,
        router: FastAPI | APIRouter,
        transport: FastApiSseTransport,
        mount_path: str,
        dependencies: Optional[Sequence[params.Depends]],
    ):
        @router.post(
            f"{mount_path}/messages/",
            include_in_schema=False,
            operation_id="mcp_messages",
            dependencies=dependencies,
        )
        async def handle_post_message(request: Request):
            return await transport.handle_fastapi_post_message(request)

    def _register_mcp_endpoints_sse(
        self,
        router: FastAPI | APIRouter,
        transport: FastApiSseTransport,
        mount_path: str,
        dependencies: Optional[Sequence[params.Depends]],
    ):
        self._register_mcp_connection_endpoint_sse(router, transport, mount_path, dependencies)
        self._register_mcp_messages_endpoint_sse(router, transport, mount_path, dependencies)

    def _register_mcp_http_endpoint(
        self,
        router: FastAPI | APIRouter,
        transport: FastApiHttpSessionManager,
        mount_path: str,
        dependencies: Optional[Sequence[params.Depends]],
    ):
        @router.api_route(
            mount_path,
            methods=["GET", "POST", "DELETE"],
            include_in_schema=False,
            operation_id="mcp_http",
            dependencies=dependencies,
        )
        async def handle_mcp_streamable_http(request: Request):
            return await transport.handle_fastapi_request(request)

    def _register_mcp_endpoints_http(
        self,
        router: FastAPI | APIRouter,
        transport: FastApiHttpSessionManager,
        mount_path: str,
        dependencies: Optional[Sequence[params.Depends]],
    ):
        self._register_mcp_http_endpoint(router, transport, mount_path, dependencies)

    def _setup_auth_2025_03_26(self):
        from fastapi_mcp.auth.proxy import (
            setup_oauth_custom_metadata,
            setup_oauth_metadata_proxy,
            setup_oauth_authorize_proxy,
            setup_oauth_fake_dynamic_register_endpoint,
        )

        if self._auth_config:
            if self._auth_config.custom_oauth_metadata:
                setup_oauth_custom_metadata(
                    app=self.fastapi,
                    auth_config=self._auth_config,
                    metadata=self._auth_config.custom_oauth_metadata,
                )

            elif self._auth_config.setup_proxies:
                assert self._auth_config.client_id is not None

                metadata_url = self._auth_config.oauth_metadata_url
                if not metadata_url:
                    metadata_url = f"{self._auth_config.issuer}{self._auth_config.metadata_path}"

                setup_oauth_metadata_proxy(
                    app=self.fastapi,
                    metadata_url=metadata_url,
                    path=self._auth_config.metadata_path,
                    register_path="/oauth/register" if self._auth_config.setup_fake_dynamic_registration else None,
                )
                setup_oauth_authorize_proxy(
                    app=self.fastapi,
                    client_id=self._auth_config.client_id,
                    authorize_url=self._auth_config.authorize_url,
                    audience=self._auth_config.audience,
                    default_scope=self._auth_config.default_scope,
                )
                if self._auth_config.setup_fake_dynamic_registration:
                    assert self._auth_config.client_secret is not None
                    setup_oauth_fake_dynamic_register_endpoint(
                        app=self.fastapi,
                        client_id=self._auth_config.client_id,
                        client_secret=self._auth_config.client_secret,
                    )

    def _setup_auth(self):
        if self._auth_config:
            if self._auth_config.version == "2025-03-26":
                self._setup_auth_2025_03_26()
            else:
                raise ValueError(
                    f"Unsupported MCP spec version: {self._auth_config.version}. Please check your AuthConfig."
                )
        else:
            logger.info("No auth config provided, skipping auth setup")

    def mount_http(
        self,
        router: Annotated[
            Optional[FastAPI | APIRouter],
            Doc(
                """
                The FastAPI app or APIRouter to mount the MCP server to. If not provided, the MCP
                server will be mounted to the FastAPI app.
                """
            ),
        ] = None,
        mount_path: Annotated[
            str,
            Doc(
                """
                Path where the MCP server will be mounted.
                Mount path is appended to the root path of FastAPI router, or to the prefix of APIRouter.
                Defaults to '/mcp'.
                """
            ),
        ] = "/mcp",
    ) -> None:
        """
        Mount the MCP server with HTTP transport to **any** FastAPI app or APIRouter.

        There is no requirement that the FastAPI app or APIRouter is the same as the one that the MCP
        server was created from.
        """
        # Normalize mount path
        if not mount_path.startswith("/"):
            mount_path = f"/{mount_path}"
        if mount_path.endswith("/"):
            mount_path = mount_path[:-1]

        if not router:
            router = self.fastapi

        assert isinstance(router, (FastAPI, APIRouter)), f"Invalid router type: {type(router)}"

        http_transport = FastApiHttpSessionManager(mcp_server=self.server)
        dependencies = self._auth_config.dependencies if self._auth_config else None

        self._register_mcp_endpoints_http(router, http_transport, mount_path, dependencies)
        self._setup_auth()
        self._http_transport = http_transport  # Store reference

        # HACK: If we got a router and not a FastAPI instance, we need to re-include the router so that
        # FastAPI will pick up the new routes we added. The problem with this approach is that we assume
        # that the router is a sub-router of self.fastapi, which may not always be the case.
        #
        # TODO: Find a better way to do this.
        if isinstance(router, APIRouter):
            self.fastapi.include_router(router)

        logger.info(f"MCP HTTP server listening at {mount_path}")

    def mount_sse(
        self,
        router: Annotated[
            Optional[FastAPI | APIRouter],
            Doc(
                """
                The FastAPI app or APIRouter to mount the MCP server to. If not provided, the MCP
                server will be mounted to the FastAPI app.
                """
            ),
        ] = None,
        mount_path: Annotated[
            str,
            Doc(
                """
                Path where the MCP server will be mounted.
                Mount path is appended to the root path of FastAPI router, or to the prefix of APIRouter.
                Defaults to '/sse'.
                """
            ),
        ] = "/sse",
    ) -> None:
        """
        Mount the MCP server with SSE transport to **any** FastAPI app or APIRouter.

        There is no requirement that the FastAPI app or APIRouter is the same as the one that the MCP
        server was created from.
        """
        # Normalize mount path
        if not mount_path.startswith("/"):
            mount_path = f"/{mount_path}"
        if mount_path.endswith("/"):
            mount_path = mount_path[:-1]

        if not router:
            router = self.fastapi

        # Build the base path correctly for the SSE transport
        assert isinstance(router, (FastAPI, APIRouter)), f"Invalid router type: {type(router)}"
        base_path = mount_path if isinstance(router, FastAPI) else router.prefix + mount_path
        messages_path = f"{base_path}/messages/"

        sse_transport = FastApiSseTransport(messages_path)
        dependencies = self._auth_config.dependencies if self._auth_config else None

        self._register_mcp_endpoints_sse(router, sse_transport, mount_path, dependencies)
        self._setup_auth()

        # HACK: If we got a router and not a FastAPI instance, we need to re-include the router so that
        # FastAPI will pick up the new routes we added. The problem with this approach is that we assume
        # that the router is a sub-router of self.fastapi, which may not always be the case.
        #
        # TODO: Find a better way to do this.
        if isinstance(router, APIRouter):
            self.fastapi.include_router(router)

        logger.info(f"MCP SSE server listening at {mount_path}")

    def mount(
        self,
        router: Annotated[
            Optional[FastAPI | APIRouter],
            Doc(
                """
                The FastAPI app or APIRouter to mount the MCP server to. If not provided, the MCP
                server will be mounted to the FastAPI app.
                """
            ),
        ] = None,
        mount_path: Annotated[
            str,
            Doc(
                """
                Path where the MCP server will be mounted.
                Mount path is appended to the root path of FastAPI router, or to the prefix of APIRouter.
                Defaults to '/mcp'.
                """
            ),
        ] = "/mcp",
        transport: Annotated[
            Literal["sse"],
            Doc(
                """
                The transport type for the MCP server. Currently only 'sse' is supported.
                This parameter is deprecated.
                """
            ),
        ] = "sse",
    ) -> None:
        """
        [DEPRECATED] Mount the MCP server to **any** FastAPI app or APIRouter.

        This method is deprecated and will be removed in a future version.
        Use mount_http() for HTTP transport (recommended) or mount_sse() for SSE transport instead.

        For backwards compatibility, this method defaults to SSE transport.

        There is no requirement that the FastAPI app or APIRouter is the same as the one that the MCP
        server was created from.
        """
        import warnings

        warnings.warn(
            "mount() is deprecated and will be removed in a future version. "
            "Use mount_http() for HTTP transport (recommended) or mount_sse() for SSE transport instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if transport == "sse":
            self.mount_sse(router, mount_path)
        else:  # pragma: no cover
            raise ValueError(  # pragma: no cover
                f"Unsupported transport: {transport}. Use mount_sse() or mount_http() instead."
            )

    def _parse_content_type(self, content_type_header: Optional[str]) -> tuple[str, Optional[str]]:
        """
        Parse a Content-Type header into media type and charset.

        Args:
            content_type_header: The Content-Type header value (e.g., "application/json; charset=utf-8")

        Returns:
            A tuple of (media_type, charset). charset may be None.
        """
        if not content_type_header:
            return "", None

        parts = content_type_header.split(";")
        media_type = parts[0].strip().lower()

        charset = None
        for part in parts[1:]:
            part = part.strip()
            if part.lower().startswith("charset="):
                charset = part[8:].strip().strip('"\'')
                break

        return media_type, charset

    def _detect_content_type_from_content(self, content: bytes) -> Optional[str]:
        """
        Detect content type by inspecting the actual content bytes.

        Args:
            content: The raw response content bytes

        Returns:
            Detected media type or None if unknown
        """
        if not content:
            return None

        # Check for common binary file signatures (magic bytes)
        # PNG
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        # JPEG
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        # GIF
        if content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
            return "image/gif"
        # WebP
        if content.startswith(b"RIFF") and len(content) > 12 and content[8:12] == b"WEBP":
            return "image/webp"
        # PDF
        if content.startswith(b"%PDF"):
            return "application/pdf"
        # ZIP (and related formats like DOCX, XLSX)
        if content.startswith(b"PK\x03\x04"):
            return "application/zip"
        # GZIP
        if content.startswith(b"\x1f\x8b"):
            return "application/gzip"

        # Try to detect text-based formats by examining content
        try:
            # Try decoding as UTF-8
            text = content.decode("utf-8").strip()

            # Check for JSON (starts with { or [)
            if text.startswith(("{", "[")):
                try:
                    json.loads(text)
                    return "application/json"
                except json.JSONDecodeError:
                    pass

            # Check for HTML first (before XML, since HTML can look like XML)
            text_lower = text.lower()
            if (
                text_lower.startswith("<!doctype html")
                or text_lower.startswith("<html")
                or ("<!doctype" in text_lower and "html" in text_lower[:100])
            ):
                return "text/html"

            # Check for XML (starts with <?xml or looks like XML but not HTML)
            if text.startswith("<?xml") or (text.startswith("<") and ">" in text and "<html" not in text_lower):
                return "application/xml"

            # If it decoded successfully and looks like text, assume plain text
            # Check if content is mostly printable
            printable_ratio = sum(1 for c in text[:1000] if c.isprintable() or c in "\n\r\t") / min(len(text), 1000)
            if printable_ratio > 0.9:
                return "text/plain"

        except (UnicodeDecodeError, ValueError):
            pass

        return None

    def _is_binary_content(self, content: bytes) -> bool:
        """
        Check if content appears to be binary (non-text) data.

        Args:
            content: The raw response content bytes

        Returns:
            True if content appears to be binary, False if it appears to be text
        """
        if not content:
            return False

        # Check for null bytes (strong indicator of binary)
        if b"\x00" in content[:1024]:
            return True

        # Try to decode as UTF-8
        try:
            text = content[:1024].decode("utf-8")
            # Check if content is mostly printable
            printable_ratio = sum(1 for c in text if c.isprintable() or c in "\n\r\t") / len(text)
            return printable_ratio < 0.7
        except UnicodeDecodeError:
            return True

    def _format_xml_for_llm(self, content: bytes) -> str:
        """
        Format XML content for better LLM readability.

        Args:
            content: The raw XML content bytes

        Returns:
            Formatted XML string with proper indentation
        """
        import xml.dom.minidom

        try:
            dom = xml.dom.minidom.parseString(content)
            # Use toprettyxml with proper settings
            formatted = dom.toprettyxml(indent="  ", encoding=None)

            # Clean up the output: remove extra blank lines and normalize whitespace
            lines = []
            for line in formatted.split("\n"):
                stripped = line.rstrip()
                # Skip empty lines but keep lines with just whitespace for structure
                if stripped or (lines and lines[-1].strip()):
                    lines.append(stripped)

            # Remove trailing empty lines
            while lines and not lines[-1].strip():
                lines.pop()

            return "\n".join(lines)
        except Exception:
            # If XML parsing fails, try to return as decoded text
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return content.decode("latin-1")

    # Maximum size for binary content to include base64 data (1MB)
    _MAX_BINARY_SIZE_FOR_BASE64 = 1024 * 1024
    # Maximum size for images to return as ImageContent (5MB)
    _MAX_IMAGE_SIZE = 5 * 1024 * 1024

    def _format_response_for_llm(
        self,
        response: httpx.Response,
    ) -> List[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
        """
        Format an HTTP response based on its content type for optimal LLM consumption.

        Uses both the Content-Type header and content inspection for robust detection.

        Args:
            response: The httpx Response object

        Returns:
            A list of MCP content types appropriate for the response
        """
        content_type_header = response.headers.get("content-type")
        header_media_type, charset = self._parse_content_type(content_type_header)
        content_length = len(response.content)

        # Handle empty responses (e.g., 204 No Content)
        if not response.content:
            return [types.TextContent(type="text", text="")]

        # Detect content type from actual content for validation/fallback
        detected_type = self._detect_content_type_from_content(response.content)

        # Determine the effective media type to use
        # Priority: header type if valid, otherwise detected type, otherwise generic
        if header_media_type:
            media_type = header_media_type
        elif detected_type:
            media_type = detected_type
        else:
            media_type = "application/octet-stream"

        # Handle JSON responses
        # Try JSON parsing if header says JSON OR if content looks like JSON
        is_json_header = media_type in ("application/json", "application/ld+json") or media_type.endswith("+json")
        is_json_detected = detected_type == "application/json"

        if is_json_header or is_json_detected:
            try:
                result = response.json()
                result_text = json.dumps(result, indent=2, ensure_ascii=False)
                return [types.TextContent(type="text", text=result_text)]
            except (json.JSONDecodeError, ValueError):
                # If header claimed JSON but parsing failed, continue to other handlers
                if is_json_header and not is_json_detected:
                    pass  # Fall through to try other content types

        # Handle image responses - check both header and magic bytes
        is_image_header = media_type.startswith("image/")
        is_image_detected = detected_type and detected_type.startswith("image/")

        if is_image_header or is_image_detected:
            # Use detected type if available (more reliable), otherwise header type
            image_media_type = detected_type if is_image_detected else media_type

            if content_length > self._MAX_IMAGE_SIZE:
                return [
                    types.TextContent(
                        type="text",
                        text=f"[Image content: {image_media_type}, {content_length:,} bytes]\n\n"
                        f"Image is too large to include inline ({content_length / (1024 * 1024):.2f} MB). "
                        f"Consider downloading separately or requesting a smaller image.",
                    )
                ]
            image_data = base64.standard_b64encode(response.content).decode("ascii")
            return [types.ImageContent(type="image", data=image_data, mimeType=image_media_type)]

        # Handle XML responses - check both header and content detection
        is_xml_header = media_type in ("application/xml", "text/xml") or media_type.endswith("+xml")
        is_xml_detected = detected_type == "application/xml"

        if is_xml_header or is_xml_detected:
            formatted_xml = self._format_xml_for_llm(response.content)
            return [types.TextContent(type="text", text=f"[XML Content]\n\n{formatted_xml}")]

        # Handle HTML responses
        is_html_header = media_type in ("text/html", "application/xhtml+xml")
        is_html_detected = detected_type == "text/html"

        if is_html_header or is_html_detected:
            try:
                html_text = response.text
            except (UnicodeDecodeError, ValueError):
                html_text = response.content.decode("utf-8", errors="replace")
            return [types.TextContent(type="text", text=f"[HTML Content]\n\n{html_text}")]

        # Handle PDF responses
        is_pdf = media_type == "application/pdf" or detected_type == "application/pdf"
        if is_pdf:
            if content_length > self._MAX_BINARY_SIZE_FOR_BASE64:
                return [
                    types.TextContent(
                        type="text",
                        text=f"[PDF Document: {content_length:,} bytes]\n\n"
                        f"PDF is too large to include inline ({content_length / (1024 * 1024):.2f} MB). "
                        f"The document should be downloaded separately for viewing.",
                    )
                ]
            binary_data = base64.standard_b64encode(response.content).decode("ascii")
            return [
                types.TextContent(
                    type="text",
                    text=f"[PDF Document: {content_length:,} bytes]\n\nBase64 encoded data:\n{binary_data}",
                )
            ]

        # Handle plain text and other text types
        is_text_header = media_type.startswith("text/")
        is_text_detected = detected_type == "text/plain"

        if is_text_header or is_text_detected:
            try:
                return [types.TextContent(type="text", text=response.text)]
            except (UnicodeDecodeError, ValueError):
                # If decoding fails, try with replacement
                return [types.TextContent(type="text", text=response.content.decode("utf-8", errors="replace"))]

        # For unknown content types, check if it's actually text
        if not self._is_binary_content(response.content):
            try:
                text = response.content.decode("utf-8")
                # Additional check: try to parse as JSON in case header was wrong
                try:
                    result = json.loads(text)
                    result_text = json.dumps(result, indent=2, ensure_ascii=False)
                    return [types.TextContent(type="text", text=result_text)]
                except json.JSONDecodeError:
                    pass
                return [types.TextContent(type="text", text=text)]
            except UnicodeDecodeError:
                pass

        # True binary content - check size and format appropriately
        if content_length > self._MAX_BINARY_SIZE_FOR_BASE64:
            return [
                types.TextContent(
                    type="text",
                    text=f"[Binary content: {media_type}, {content_length:,} bytes]\n\n"
                    f"Content is too large to include inline ({content_length / (1024 * 1024):.2f} MB). "
                    f"The response contains binary data that should be downloaded separately.",
                )
            ]

        # Small binary: encode as base64 and include
        binary_data = base64.standard_b64encode(response.content).decode("ascii")
        return [
            types.TextContent(
                type="text",
                text=f"[Binary content: {media_type}, {content_length:,} bytes]\n\nBase64 encoded data:\n{binary_data}",
            )
        ]

    async def _execute_api_tool(
        self,
        client: Annotated[httpx.AsyncClient, Doc("httpx client to use in API calls")],
        tool_name: Annotated[str, Doc("The name of the tool to execute")],
        arguments: Annotated[Dict[str, Any], Doc("The arguments for the tool")],
        operation_map: Annotated[Dict[str, Dict[str, Any]], Doc("A mapping from tool names to operation details")],
        http_request_info: Annotated[
            Optional[HTTPRequestInfo],
            Doc("HTTP request info to forward to the actual API call"),
        ] = None,
    ) -> List[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
        """
        Execute an MCP tool by making an HTTP request to the corresponding API endpoint.

        Returns:
            The result as MCP content types
        """
        if tool_name not in operation_map:
            raise Exception(f"Unknown tool: {tool_name}")

        operation = operation_map[tool_name]
        path: str = operation["path"]
        method: str = operation["method"]
        parameters: List[Dict[str, Any]] = operation.get("parameters", [])
        arguments = arguments.copy() if arguments else {}  # Deep copy arguments to avoid mutating the original

        for param in parameters:
            if param.get("in") == "path" and param.get("name") in arguments:
                param_name = param.get("name", None)
                if param_name is None:
                    raise ValueError(f"Parameter name is None for parameter: {param}")
                path = path.replace(f"{{{param_name}}}", str(arguments.pop(param_name)))

        query = {}
        for param in parameters:
            if param.get("in") == "query" and param.get("name") in arguments:
                param_name = param.get("name", None)
                if param_name is None:
                    raise ValueError(f"Parameter name is None for parameter: {param}")
                query[param_name] = arguments.pop(param_name)

        headers = {}
        for param in parameters:
            if param.get("in") == "header" and param.get("name") in arguments:
                param_name = param.get("name", None)
                if param_name is None:
                    raise ValueError(f"Parameter name is None for parameter: {param}")
                headers[param_name] = arguments.pop(param_name)

        # Forward headers that are in the allowlist
        if http_request_info and http_request_info.headers:
            for name, value in http_request_info.headers.items():
                # case-insensitive check for allowed headers
                if name.lower() in self._forward_headers:
                    headers[name] = value

        body = arguments if arguments else None

        try:
            logger.debug(f"Making {method.upper()} request to {path}")
            response = await self._request(client, method, path, query, headers, body)

            # Check for HTTP errors before processing response
            # TODO: Use a raise_for_status() method on the response (it needs to also be implemented in the AsyncClientProtocol)
            if 400 <= response.status_code < 600:
                raise Exception(
                    f"Error calling {tool_name}. Status code: {response.status_code}. Response: {response.text}"
                )

            # Format response based on content type for optimal LLM consumption
            return self._format_response_for_llm(response)

        except Exception as e:
            logger.exception(f"Error calling {tool_name}")
            raise e

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        query: Dict[str, Any],
        headers: Dict[str, str],
        body: Optional[Any],
    ) -> Any:
        if method.lower() == "get":
            return await client.get(path, params=query, headers=headers)
        elif method.lower() == "post":
            return await client.post(path, params=query, headers=headers, json=body)
        elif method.lower() == "put":
            return await client.put(path, params=query, headers=headers, json=body)
        elif method.lower() == "delete":
            return await client.delete(path, params=query, headers=headers)
        elif method.lower() == "patch":
            return await client.patch(path, params=query, headers=headers, json=body)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

    def _filter_tools(self, tools: List[types.Tool], openapi_schema: Dict[str, Any]) -> List[types.Tool]:
        """
        Filter tools based on operation IDs and tags.

        Args:
            tools: List of tools to filter
            openapi_schema: The OpenAPI schema

        Returns:
            Filtered list of tools
        """
        if (
            self._include_operations is None
            and self._exclude_operations is None
            and self._include_tags is None
            and self._exclude_tags is None
        ):
            return tools

        operations_by_tag: Dict[str, List[str]] = {}
        for path, path_item in openapi_schema.get("paths", {}).items():
            for method, operation in path_item.items():
                if method not in ["get", "post", "put", "delete", "patch"]:
                    continue

                operation_id = operation.get("operationId")
                if not operation_id:
                    continue

                tags = operation.get("tags", [])
                for tag in tags:
                    if tag not in operations_by_tag:
                        operations_by_tag[tag] = []
                    operations_by_tag[tag].append(operation_id)

        operations_to_include = set()

        if self._include_operations is not None:
            operations_to_include.update(self._include_operations)
        elif self._exclude_operations is not None:
            all_operations = {tool.name for tool in tools}
            operations_to_include.update(all_operations - set(self._exclude_operations))

        if self._include_tags is not None:
            for tag in self._include_tags:
                operations_to_include.update(operations_by_tag.get(tag, []))
        elif self._exclude_tags is not None:
            excluded_operations = set()
            for tag in self._exclude_tags:
                excluded_operations.update(operations_by_tag.get(tag, []))

            all_operations = {tool.name for tool in tools}
            operations_to_include.update(all_operations - excluded_operations)

        filtered_tools = [tool for tool in tools if tool.name in operations_to_include]

        if filtered_tools:
            filtered_operation_ids = {tool.name for tool in filtered_tools}
            self.operation_map = {
                op_id: details for op_id, details in self.operation_map.items() if op_id in filtered_operation_ids
            }

        return filtered_tools
