"""
Model Context Protocol (MCP) Server Connectors for smolagents
Provides pre-configured MCP servers for common tools
"""

import os
from typing import List, Optional
from smolagents import ToolCollection
from mcp import StdioServerParameters

# MCP Server Configurations
MCP_SERVERS = {
    "filesystem": {
        "description": "Read, write, and manage local files and directories",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", os.path.expanduser("~")],
    },
    "github": {
        "description": "GitHub repository operations (search, issues, PRs)",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_vars": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
    },
    "brave_search": {
        "description": "Web search using Brave Search API",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env_vars": ["BRAVE_API_KEY"],
    },
    "slack": {
        "description": "Slack workspace operations (channels, messages, users)",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env_vars": ["SLACK_BOT_TOKEN"],
    },
    "google_maps": {
        "description": "Google Maps location and directions",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-google-maps"],
        "env_vars": ["GOOGLE_MAPS_API_KEY"],
    },
    "postgres": {
        "description": "PostgreSQL database operations",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres"],
        "env_vars": ["POSTGRES_CONNECTION_STRING"],
    },
    "pubmed": {
        "description": "Search biomedical literature via PubMed",
        "command": "uvx",
        "args": ["--quiet", "pubmedmcp@0.1.3"],
        "env_vars": [],
    },
}


def get_available_mcp_servers() -> List[str]:
    """Get list of available MCP server names"""
    return list(MCP_SERVERS.keys())


def get_mcp_server_info(server_name: str) -> Optional[dict]:
    """Get information about a specific MCP server"""
    return MCP_SERVERS.get(server_name)


def check_mcp_server_requirements(server_name: str) -> tuple[bool, List[str]]:
    """
    Check if required environment variables are set for a server

    Returns:
        Tuple of (all_requirements_met, list_of_missing_vars)
    """
    server_config = MCP_SERVERS.get(server_name)
    if not server_config:
        return False, [f"Unknown server: {server_name}"]

    required_vars = server_config.get("env_vars", [])
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    return len(missing_vars) == 0, missing_vars


def load_mcp_server(
    server_name: str,
    trust_remote_code: bool = True,
    structured_output: bool = False
) -> Optional[ToolCollection]:
    """
    Load an MCP server as a ToolCollection

    Args:
        server_name: Name of the MCP server to load
        trust_remote_code: Whether to trust remote code execution
        structured_output: Enable structured output features

    Returns:
        ToolCollection if successful, None otherwise
    """
    server_config = MCP_SERVERS.get(server_name)
    if not server_config:
        raise ValueError(f"Unknown MCP server: {server_name}. Available: {list(MCP_SERVERS.keys())}")

    requirements_met, missing_vars = check_mcp_server_requirements(server_name)
    if not requirements_met:
        raise ValueError(
            f"Missing required environment variables for {server_name}: {', '.join(missing_vars)}"
        )

    env = {**os.environ}

    if server_config["command"] == "uvx":
        env["UV_PYTHON"] = "3.12"

    server_params = StdioServerParameters(
        command=server_config["command"],
        args=server_config["args"],
        env=env
    )

    try:
        tool_collection = ToolCollection.from_mcp(
            server_parameters=server_params,
            trust_remote_code=trust_remote_code,
            structured_output=structured_output
        )
        return tool_collection
    except Exception as e:
        raise RuntimeError(f"Failed to load MCP server '{server_name}': {e}")


def load_multiple_mcp_servers(
    server_names: List[str],
    trust_remote_code: bool = True,
    structured_output: bool = False
) -> List[ToolCollection]:
    """
    Load multiple MCP servers at once

    Args:
        server_names: List of server names to load
        trust_remote_code: Whether to trust remote code execution
        structured_output: Enable structured output features

    Returns:
        List of ToolCollections (skips servers that fail to load)
    """
    tool_collections = []

    for server_name in server_names:
        try:
            collection = load_mcp_server(
                server_name,
                trust_remote_code=trust_remote_code,
                structured_output=structured_output
            )
            tool_collections.append(collection)
            print(f"✓ Loaded MCP server: {server_name}")
        except Exception as e:
            print(f"✗ Failed to load {server_name}: {e}")

    return tool_collections
