#!/usr/bin/env python

"""
title: MySQL MCP Tool 
author: Dhiraj
description: MySQL MCP tool to run read-only queries
licence: MIT
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Union
from contextlib import contextmanager

from fastmcp import FastMCP
from mysql.connector import Error, pooling
from dotenv import load_dotenv

load_dotenv()

# ---- JSON Logging setup ----
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("B2CC-MYSQL-MCP")

# --- Configuration ---
MYSQL_HOST = os.environ["DB_HOST"]
MYSQL_USER = os.environ["DB_USER"]
MYSQL_PASSWORD = os.environ["DB_PASSWORD"]
MYSQL_DB = os.environ["DB_NAME"]
MCP_PORT = int(os.environ.get("MCP_PORT", 9001))

MAX_ROWS = int(os.environ.get("MAX_ROWS", 100))  # limit rows per query

# --- Connection Pool ---
try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="b2cc-pool",
        pool_size=5,
        pool_reset_session=True,
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        connection_timeout=10,
    )
    logger.info("MySQL connection pool created successfully.")
except Error as e:
    logger.error(f"Error creating connection pool: {e}")
    raise

@contextmanager
def get_db_cursor(dictionary: bool = False):
    """Context manager to provide a pooled DB cursor with safe cleanup."""
    connection = None
    cursor = None
    try:
        connection = connection_pool.get_connection()
        cursor = connection.cursor(dictionary=dictionary)
        yield cursor
    except Error as e:
        logger.error(f"Database error: {e}")
        yield None
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# --- FastMCP Initialization ---
mcp = FastMCP("B2CC MySQL Tool Server 🚀")

# --- Cached Schema ---
DB_SCHEMA: Dict[str, Any] = {}

def load_all_table_schemas() -> Dict[str, Any]:
    """Loads schema for all tables at startup and caches it."""
    global DB_SCHEMA
    with get_db_cursor() as cursor:
        if not cursor:
            return {}
        try:
            cursor.execute("SHOW TABLES")
            tables = [t[0] for t in cursor.fetchall()]
            schemas = {}
            for table in tables:
                cursor.execute(f"DESCRIBE `{table}`")
                schemas[table] = [
                    {
                        "Field": row[0],
                        "Type": row[1],
                        "Null": row[2],
                        "Key": row[3],
                        "Default": row[4],
                        "Extra": row[5],
                    }
                    for row in cursor.fetchall()
                ]
            DB_SCHEMA = schemas
            logger.info("Database schemas cached successfully.")
            return schemas
        except Error as e:
            logger.error(f"Error retrieving table schemas: {e}")
            return {}

# --- Query Execution ---
def run_single_query(cursor, sql_query: str) -> Dict[str, Any]:
    """Executes a single SELECT query with row limiting."""
    if not sql_query.lower().strip().startswith("select"):
        return {"success": False, "data": None, "error": "Only SELECT queries are allowed."}

    # enforce LIMIT if missing
    safe_query = sql_query.strip().rstrip(";")
    if "limit" not in safe_query.lower():
        safe_query += f" LIMIT {MAX_ROWS}"

    start_time = time.time()
    try:
        cursor.execute(safe_query)
        result = cursor.fetchall()
        duration = time.time() - start_time
        logger.info(f"Executed query in {duration:.2f}s: {safe_query}")
        return {"success": True, "data": result, "error": None}
    except Error as e:
        logger.error(f"Query failed: {safe_query} | Error: {e}")
        return {"success": False, "data": None, "error": "Database query failed."}

# --- Tools ---
@mcp.tool
def execute_sql_query(sql_query: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Executes one or more SQL SELECT queries and returns structured results.
    Automatically limits rows to MAX_ROWS per query.
    """
    with get_db_cursor(dictionary=True) as cursor:
        if not cursor:
            return {"success": False, "data": None, "error": "Database connection failed."}

        if isinstance(sql_query, str):
            return run_single_query(cursor, sql_query)

        if isinstance(sql_query, list):
            results = []
            for q in sql_query:
                results.append(run_single_query(cursor, q))
            return {"success": True, "data": results, "error": None}

        return {"success": False, "data": None, "error": "Invalid query type. Must be string or list of strings."}

@mcp.tool
def list_tables() -> Dict[str, Any]:
    """Lists all tables in the connected database."""
    return {"success": True, "data": list(DB_SCHEMA.keys()), "error": None}

@mcp.tool
def get_table_schema(table_name: str) -> Dict[str, Any]:
    """Returns the schema for a single table from cached metadata."""
    schema = DB_SCHEMA.get(table_name)
    if not schema:
        return {"success": False, "data": None, "error": f"No schema found for table {table_name}"}
    return {"success": True, "data": schema, "error": None}

@mcp.tool
def get_all_table_schemas() -> Dict[str, Any]:
    """Returns cached schemas for all tables."""
    if not DB_SCHEMA:
        return {"success": False, "data": None, "error": "Schema cache is empty."}
    return {"success": True, "data": DB_SCHEMA, "error": None}

# --- Entry Point ---
if __name__ == "__main__":
    if not all([MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB]):
        print("Error: Missing required MySQL environment variables.")
    else:
        load_all_table_schemas()  # cache schema at startup
        mcp.run(transport="http", host="0.0.0.0", port=MCP_PORT)