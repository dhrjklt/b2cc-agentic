#!/usr/bin/env python

"""
title: GitLab MCP Tool 
author: Dhiraj
description: GitLab MCP tool to fetch tickets and documentation files
licence: MIT
"""

import os
import logging
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from fastmcp import FastMCP
import gitlab

# --- Load Environment Variables ---
load_dotenv()
GITLAB_URL = os.environ.get("GITLAB_URL", "https://gitlab.desy.de")
GITLAB_PRIVATE_TOKEN = os.environ.get("GITLAB_PRIVATE_TOKEN", "")
PROJECT_ID = os.environ.get("PROJECT_ID")
MCP_PORT = int(os.environ.get("MCP_PORT", 9002))

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- FastMCP Initialization ---
mcp = FastMCP("DESY GitLab Tool Server 🚀")

# --- GitLab Connection ---
def connect_gitlab():
    """Connects to GitLab and returns the project object."""
    if not all([GITLAB_URL, GITLAB_PRIVATE_TOKEN, PROJECT_ID]):
        logger.error("Please set GITLAB_URL, GITLAB_PRIVATE_TOKEN, and PROJECT_ID environment variables.")
        return None

    try:
        gl = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_PRIVATE_TOKEN)
        gl.auth()
        project = gl.projects.get(PROJECT_ID)
        logger.info("Successfully connected to GitLab.")
        return project
    except Exception as e:
        logger.error(f"Could not connect to GitLab. Error: {e}")
        return None

project = connect_gitlab()

# --- MCP Tools ---

@mcp.tool(description="Get details of a specific ticket (issue) by its ID.")
def get_ticket_details(ticket_id: int) -> Dict[str, Any]:
    if ticket_id <= 0:
        return {"error": "Invalid ticket_id"}
    if not project:
        return {"error": "GitLab connection not available."}
    try:
        issue = project.issues.get(ticket_id)
        return issue.asdict()
    except gitlab.exceptions.GitlabGetError as e:
        if e.response_code == 404:
            return {"error": f"Ticket with ID {ticket_id} not found."}
        return {"error": f"GitLab API error: {e}"}
    except Exception as e:
        return {"error": str(e)}

# @mcp.tool(description="Get the content of a specific documentation file by its path.")
# def get_documentation_file(file_path: str) -> Dict[str, str]:
#     if not project:
#         return {"error": "GitLab connection not available."}
#     try:
#         ref = project.default_branch
#         file_content = project.files.get(file_path=file_path, ref=ref)
#         return {"file_path": file_path, "content": file_content.decode('utf-8')}
#     except gitlab.exceptions.GitlabGetError:
#         return {"error": f"Documentation file not found at path: {file_path}"}
#     except Exception as e:
#         return {"error": str(e)}

# @mcp.tool(description="List all documentation file paths in a specified directory of the repository.")
# def list_documentation_files(path: str = 'docs') -> Union[List[Dict[str, str]], Dict[str, str]]:
#     if not project:
#         return {"error": "GitLab connection not available."}
#     try:
#         ref = project.default_branch
#         items = project.repository_tree(path=path, ref=ref, recursive=True, all=True)
#         return [item for item in items if item['type'] == 'blob']
#     except Exception as e:
#         return {"error": str(e)}

@mcp.tool(description="Get all tickets (issues) in the project, with optional filtering by state.")
def get_all_tickets(state: str = "all", max_results: int = 1000) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
    Retrieves all issues in the project with pagination.

    Args:
        state (str): Filter by 'open', 'close', or 'all'. Default is 'all'.
        max_results (int): Maximum number of issues to fetch.

    Returns:
        List of issues as dictionaries or error dict.
    """
    if not project:
        return {"error": "GitLab connection not available."}

    if state not in ["opened", "closed", "all"]:
        return {"error": "Invalid state. Use 'opened', 'closed', or 'all'."}

    try:
        issues = []
        page = 1
        per_page = 100  # GitLab max per page
        while True:
            batch = project.issues.list(state=state, page=page, per_page=per_page)
            if not batch:
                break
            issues.extend([issue.asdict() for issue in batch])
            if len(issues) >= max_results:
                break
            page += 1

        return issues[:max_results]
    except Exception as e:
        return {"error": str(e)}

# --- Run MCP Server ---
if __name__ == "__main__":
    if project:
        mcp.run(transport="http", host="0.0.0.0", port=MCP_PORT)
    else:
        logger.error("Halting MCP server due to failed GitLab connection.")