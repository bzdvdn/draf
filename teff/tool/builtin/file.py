"""File tools — read, write, and edit files on disk."""

from teff.tool.tool import Tool


class ReadFileTool(Tool):
    """Read a file's contents."""

    name = "read_file"
    description = "Read a file"

    def run(self, path: str = "") -> str:  # type: ignore[override]
        with open(path) as f:
            return f.read()


class WriteFileTool(Tool):
    """Write text content to a file."""

    name = "write_file"
    description = "Write content to a file"

    def run(self, path: str = "", content: str = "") -> str:  # type: ignore[override]
        with open(path, "w") as f:
            f.write(content)
        return f"written {len(content)} bytes to {path}"


class EditFileTool(Tool):
    """Edit a file by replacing the first occurrence of a string."""

    name = "edit_file"
    description = "Edit a file by replacing text"

    def run(self, path: str = "", old: str = "", new: str = "") -> str:  # type: ignore[override]
        with open(path) as f:
            content = f.read()
        if old not in content:
            raise ValueError(f"text not found in {path}")
        content = content.replace(old, new, 1)
        with open(path, "w") as f:
            f.write(content)
        return f"replaced in {path}"
