"""SQL tools — read-only queries and schema inspection for SQLite/PostgreSQL."""

from teff.tool.tool import Tool

_SQLITE_TABLES = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
_POSTGRES_TABLES = (
    "SELECT table_name FROM information_schema.tables "
    "WHERE table_schema='public' ORDER BY table_name"
)


class _SQLBase(Tool):
    """Shared connection setup for SQL tools.

    Supported backends:

    - ``sqlite`` (default, stdlib): config key ``path`` → database file.
    - ``postgres``: config key ``dsn`` → a libpq-style connection string
      (e.g. ``postgresql://user:pass@host:5432/db``). Requires ``psycopg``
      (from ``teff[tools]``).

    Args:
        config: Optional dict with ``db_type`` ("sqlite"|"postgres"),
            ``path`` (SQLite file), ``dsn`` (PostgreSQL connection string).
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.db_type = cfg.get("db_type", "sqlite")
        self.path = cfg.get("path", "")
        self.dsn = cfg.get("dsn", "")

    def _connect(self):
        if self.db_type == "sqlite":
            if not self.path:
                raise ValueError("requires 'path' in config for db_type=sqlite")
            import sqlite3

            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            return conn
        if self.db_type == "postgres":
            if not self.dsn:
                raise ValueError("requires 'dsn' in config for db_type=postgres")
            try:
                import psycopg
            except ImportError as e:
                msg = "postgres tools require 'psycopg' (pip install teff[tools])"
                raise ImportError(msg) from e
            return psycopg.connect(self.dsn)
        msg = f"unsupported db_type: {self.db_type}"
        raise ValueError(msg)

    @staticmethod
    def _format(description, rows) -> str:
        columns = [col[0] for col in (description or [])]
        header = "\t".join(columns) if columns else ""
        lines = [header] if header else []
        for row in rows:
            lines.append("\t".join("NULL" if v is None else str(v) for v in row))
        return "\n".join(lines) if lines else "no rows"


class SQLQueryTool(_SQLBase):
    """Run a read-only SQL query against a database.

    Only ``SELECT``/``WITH`` (read) statements are allowed; anything that
    would mutate data (``INSERT``, ``UPDATE``, ``DELETE``, DDL, …) is
    rejected. Placeholders match the backend: ``?`` for SQLite, ``%s``
    for PostgreSQL.
    """

    name = "sql_query"
    description = "Run a read-only SQL SELECT query against a database"

    def _guard(self, query: str) -> None:
        first = query.lstrip().split(None, 1)[0].upper() if query.strip() else ""
        if first not in ("SELECT", "WITH", "EXPLAIN"):
            msg = f"sql_query is read-only: unsupported statement '{first or query}'"
            raise ValueError(msg)

    def run(self, query: str = "", params: list | None = None, limit: int = 100) -> str:  # type: ignore[override]
        if not query:
            raise ValueError("query is required")
        self._guard(query)
        conn = self._connect()
        try:
            if self.db_type == "sqlite":
                cursor = conn.execute(query, params or ())
            else:
                with conn.cursor() as cursor:
                    cursor.execute(query, params or ())
                    rows = cursor.fetchmany(limit)
                    return self._format(cursor.description, rows)
            return self._format(cursor.description, cursor.fetchmany(limit))
        finally:
            conn.close()


class SQLListTablesTool(_SQLBase):
    """List the tables in a database."""

    name = "sql_list_tables"
    description = "List the tables in a database"

    def run(self) -> str:  # type: ignore[override]
        conn = self._connect()
        try:
            if self.db_type == "sqlite":
                cursor = conn.execute(_SQLITE_TABLES)
            else:
                with conn.cursor() as cursor:
                    cursor.execute(_POSTGRES_TABLES)
                    return self._format(cursor.description, cursor.fetchall())
            return self._format(cursor.description, cursor.fetchall())
        finally:
            conn.close()


class SQLDescribeTool(_SQLBase):
    """Describe a table's columns and types."""

    name = "sql_describe"
    description = "Describe a table's columns and types"

    def run(self, table: str = "") -> str:  # type: ignore[override]
        if not table:
            raise ValueError("table is required")
        conn = self._connect()
        try:
            if self.db_type == "sqlite":
                cursor = conn.execute(f'PRAGMA table_info("{table}")')
                return self._format(cursor.description, cursor.fetchall())
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name=%s "
                    "ORDER BY ordinal_position",
                    (table,),
                )
                return self._format(cursor.description, cursor.fetchall())
        finally:
            conn.close()
