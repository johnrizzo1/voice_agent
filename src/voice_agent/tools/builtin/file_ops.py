"""
File operations tool for basic file system tasks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..base import Tool


class FileOpsParameters(BaseModel):
    """Parameters for the file operations tool."""

    operation: str = Field(
        description="Operation to perform. Supports both exact operations (read, write, list, exists, delete) and natural language (save, create, open, view, etc.)"
    )
    path: str = Field(description="File or directory path")
    content: str = Field(default="", description="Content for write operations")
    recursive: bool = Field(
        default=False, description="Recursive operation for directory listing"
    )


class FileOpsTool(Tool):
    """
    File operations tool for basic file system tasks.

    Supports reading, writing, listing, and checking file existence.
    Includes safety measures to prevent unauthorized access.
    """

    name = "file_ops"
    description = "Perform basic file system operations"
    version = "1.0.0"

    Parameters = FileOpsParameters

    # Allowed file extensions for safety
    ALLOWED_EXTENSIONS = {
        ".txt",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".log",
        ".py",
        ".js",
        ".html",
        ".css",
        ".xml",
        ".ini",
        ".cfg",
    }

    # Forbidden paths to prevent system access
    FORBIDDEN_PATHS = {
        "/etc",
        "/sys",
        "/proc",
        "/dev",
        "/boot",
        "/root",
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\System32",
    }

    def __init__(self):
        """Initialize the file operations tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Set working directory to user's home by default
        self.base_directory = Path.home()

    def execute(
        self, operation: str, path: str, content: str = "", recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a file operation.

        Args:
            operation: Operation to perform
            path: File or directory path
            content: Content for write operations
            recursive: Recursive operation for directory listing

        Returns:
            Dictionary containing operation result
        """
        try:
            # Validate and normalize the path
            file_path = self._validate_path(path)
            if not file_path:
                return {
                    "success": False,
                    "result": None,
                    "error": "Invalid or forbidden path",
                }

            # Map natural language operations to supported operations
            normalized_operation = self._normalize_operation(operation)

            # Execute the requested operation
            if normalized_operation == "read":
                return self._read_file(file_path)
            elif normalized_operation == "write":
                return self._write_file(file_path, content)
            elif normalized_operation == "list":
                return self._list_directory(file_path, recursive)
            elif normalized_operation == "exists":
                return self._check_exists(file_path)
            elif normalized_operation == "delete":
                return self._delete_file(file_path)
            else:
                supported_ops = ["read", "write", "list", "exists", "delete"]
                natural_language_examples = [
                    "save/create/store → write",
                    "open/view → read",
                    "show/display → list",
                    "check → exists",
                    "remove → delete",
                ]
                return {
                    "success": False,
                    "result": None,
                    "error": f"Unknown operation: '{operation}'. Supported operations: {', '.join(supported_ops)}. Natural language examples: {'; '.join(natural_language_examples)}",
                }

        except Exception as e:
            self.logger.error(f"File operation error: {e}")
            return {"success": False, "result": None, "error": str(e)}

    def _normalize_operation(self, operation: str) -> str:
        """
        Normalize natural language operations to supported operations.

        Args:
            operation: Raw operation string from user

        Returns:
            Normalized operation string or original if no mapping found
        """
        # Convert to lowercase for case-insensitive matching
        op_lower = operation.lower().strip()

        # Operation mappings from natural language to supported operations
        operation_mappings = {
            # Write operation mappings
            "save": "write",
            "create": "write",
            "store": "write",
            "write to": "write",
            "save to": "write",
            "create file": "write",
            "save file": "write",
            "write file": "write",
            "store in": "write",
            "put": "write",
            # Read operation mappings
            "open": "read",
            "view": "read",
            "read from": "read",
            "open file": "read",
            "view file": "read",
            "show": "read",
            "display": "read",
            "get": "read",
            "load": "read",
            # List operation mappings
            "list": "list",
            "show directory": "list",
            "list directory": "list",
            "list files": "list",
            "show files": "list",
            "dir": "list",
            "ls": "list",
            # Exists operation mappings
            "exists": "exists",
            "check": "exists",
            "check if exists": "exists",
            "verify": "exists",
            "find": "exists",
            # Delete operation mappings
            "delete": "delete",
            "remove": "delete",
            "rm": "delete",
            "del": "delete",
            "erase": "delete",
        }

        # Check for direct mapping
        if op_lower in operation_mappings:
            return operation_mappings[op_lower]

        # Check for partial matches (e.g., "save this to file" contains "save")
        for natural_op, normalized_op in operation_mappings.items():
            if natural_op in op_lower:
                return normalized_op

        # If no mapping found, return the original operation
        return op_lower

    def _validate_path(self, path: str) -> Path:
        """
        Validate and normalize a file path for security.

        Args:
            path: Raw path string

        Returns:
            Validated Path object or None if invalid
        """
        try:
            # Convert to Path object
            file_path = Path(path)

            # If path is relative, make it relative to base directory
            if not file_path.is_absolute():
                file_path = self.base_directory / file_path

            # Resolve to absolute path to handle .. and .
            file_path = file_path.resolve()

            # Check if path is within allowed areas
            path_str = str(file_path)
            for forbidden in self.FORBIDDEN_PATHS:
                if path_str.startswith(forbidden):
                    self.logger.warning(
                        f"Access to forbidden path attempted: {path_str}"
                    )
                    return None

            # Check file extension for write operations
            if (
                file_path.suffix
                and file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS
            ):
                self.logger.warning(
                    f"Access to non-allowed file type: {file_path.suffix}"
                )
                return None

            return file_path

        except Exception as e:
            self.logger.error(f"Path validation error: {e}")
            return None

    def _read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read contents of a file."""
        try:
            if not file_path.exists():
                return {
                    "success": False,
                    "result": None,
                    "error": "File does not exist",
                }

            if not file_path.is_file():
                return {"success": False, "result": None, "error": "Path is not a file"}

            # Read file with UTF-8 encoding
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "success": True,
                "result": {
                    "content": content,
                    "path": str(file_path),
                    "size": len(content),
                    "lines": len(content.splitlines()),
                },
                "error": None,
            }

        except UnicodeDecodeError:
            return {
                "success": False,
                "result": None,
                "error": "File contains non-UTF-8 content",
            }
        except PermissionError:
            return {"success": False, "result": None, "error": "Permission denied"}

    def _write_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file with UTF-8 encoding
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "result": {
                    "path": str(file_path),
                    "size": len(content),
                    "lines": len(content.splitlines()),
                    "message": "File written successfully",
                },
                "error": None,
            }

        except PermissionError:
            return {"success": False, "result": None, "error": "Permission denied"}

    def _list_directory(
        self, dir_path: Path, recursive: bool = False
    ) -> Dict[str, Any]:
        """List contents of a directory."""
        try:
            if not dir_path.exists():
                return {
                    "success": False,
                    "result": None,
                    "error": "Directory does not exist",
                }

            if not dir_path.is_dir():
                return {
                    "success": False,
                    "result": None,
                    "error": "Path is not a directory",
                }

            items = []

            if recursive:
                # Recursive listing
                for item in dir_path.rglob("*"):
                    if item.is_file():
                        items.append(
                            {
                                "name": item.name,
                                "path": str(item.relative_to(dir_path)),
                                "type": "file",
                                "size": item.stat().st_size,
                            }
                        )
                    elif item.is_dir():
                        items.append(
                            {
                                "name": item.name,
                                "path": str(item.relative_to(dir_path)),
                                "type": "directory",
                                "size": None,
                            }
                        )
            else:
                # Non-recursive listing
                for item in dir_path.iterdir():
                    if item.is_file():
                        items.append(
                            {
                                "name": item.name,
                                "type": "file",
                                "size": item.stat().st_size,
                            }
                        )
                    elif item.is_dir():
                        items.append(
                            {"name": item.name, "type": "directory", "size": None}
                        )

            return {
                "success": True,
                "result": {
                    "path": str(dir_path),
                    "items": sorted(items, key=lambda x: (x["type"], x["name"])),
                    "total_items": len(items),
                    "recursive": recursive,
                },
                "error": None,
            }

        except PermissionError:
            return {"success": False, "result": None, "error": "Permission denied"}

    def _check_exists(self, file_path: Path) -> Dict[str, Any]:
        """Check if a file or directory exists."""
        try:
            exists = file_path.exists()

            result = {"path": str(file_path), "exists": exists}

            if exists:
                result.update(
                    {
                        "type": "file" if file_path.is_file() else "directory",
                        "size": (
                            file_path.stat().st_size if file_path.is_file() else None
                        ),
                    }
                )

            return {"success": True, "result": result, "error": None}

        except PermissionError:
            return {"success": False, "result": None, "error": "Permission denied"}

    def _delete_file(self, file_path: Path) -> Dict[str, Any]:
        """Delete a file."""
        try:
            if not file_path.exists():
                return {
                    "success": False,
                    "result": None,
                    "error": "File does not exist",
                }

            if file_path.is_dir():
                return {
                    "success": False,
                    "result": None,
                    "error": "Cannot delete directories (safety measure)",
                }

            # Delete the file
            file_path.unlink()

            return {
                "success": True,
                "result": {
                    "path": str(file_path),
                    "message": "File deleted successfully",
                },
                "error": None,
            }

        except PermissionError:
            return {"success": False, "result": None, "error": "Permission denied"}

    def get_allowed_extensions(self) -> List[str]:
        """Get list of allowed file extensions."""
        return sorted(list(self.ALLOWED_EXTENSIONS))

    def set_base_directory(self, directory: str) -> bool:
        """
        Set the base directory for relative paths.

        Args:
            directory: Base directory path

        Returns:
            True if successful, False otherwise
        """
        try:
            new_base = Path(directory).resolve()
            if new_base.exists() and new_base.is_dir():
                self.base_directory = new_base
                self.logger.info(f"Base directory set to: {new_base}")
                return True
            else:
                self.logger.warning(f"Invalid base directory: {directory}")
                return False
        except Exception as e:
            self.logger.error(f"Error setting base directory: {e}")
            return False

    def get_help(self) -> Dict[str, Any]:
        """Get help information for the file operations tool."""
        return {
            "name": self.name,
            "description": self.description,
            "operations": {
                "read": "Read contents of a file (also: open, view, show, display, get, load)",
                "write": "Write content to a file (also: save, create, store, put)",
                "list": "List directory contents (also: show directory, dir, ls)",
                "exists": "Check if file/directory exists (also: check, verify, find)",
                "delete": "Delete a file (also: remove, rm, del, erase)",
            },
            "parameters": {
                "operation": "Operation to perform",
                "path": "File or directory path",
                "content": "Content for write operations",
                "recursive": "Recursive listing for directories",
            },
            "allowed_extensions": self.get_allowed_extensions(),
            "base_directory": str(self.base_directory),
            "safety_notes": [
                "Access is restricted to allowed file types",
                "System directories are forbidden",
                "Directory deletion is not allowed",
                "All paths are validated for security",
            ],
        }
