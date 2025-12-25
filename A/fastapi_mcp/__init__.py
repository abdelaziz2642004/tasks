"""
FastAPI-MCP: Automatic MCP server generator for FastAPI applications.

Created by Tadata Inc. (https://github.com/tadata-org)
"""

try:
    from importlib.metadata import version

    __version__ = version("fastapi-mcp")
except Exception:  # pragma: no cover
    # Fallback for local development
    __version__ = "0.0.0.dev0"  # pragma: no cover

from .server import FastApiMCP
from .types import AuthConfig, OAuthMetadata
from .openapi.utils import (
    ReferenceResolutionContext,
    ReferenceResolutionResult,
    SchemaAnalysisResult,
    UnresolvedReferenceError,
    resolve_schema_references_with_details,
    analyze_schema_references,
    validate_resolved_schema,
)


__all__ = [
    "FastApiMCP",
    "AuthConfig",
    "OAuthMetadata",
    "ReferenceResolutionContext",
    "ReferenceResolutionResult",
    "SchemaAnalysisResult",
    "UnresolvedReferenceError",
    "resolve_schema_references_with_details",
    "analyze_schema_references",
    "validate_resolved_schema",
]
