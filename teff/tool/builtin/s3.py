"""S3 tools — list, read, and write objects in Amazon S3 (or S3-compatible stores)."""

from teff.tool.tool import Tool


class S3Tool(Tool):
    """List objects in an S3 bucket.

    Requires ``boto3`` (from ``teff[tools]``). Credentials are resolved by
    boto3's standard chain (env vars, ``~/.aws/credentials``, IAM role)
    unless overridden via *config*.

    Args:
        config: Optional dict with ``bucket``, ``region``,
            ``endpoint_url`` (for S3-compatible stores like MinIO),
            ``aws_access_key_id``, ``aws_secret_access_key``, ``verify``
            (TLS verification: ``True``/``False`` or a CA bundle path;
            set ``False`` for self-signed local endpoints).
    """

    name = "s3_list"
    description = "List objects in an S3 bucket"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.bucket = cfg.get("bucket", "")
        self.region = cfg.get("region")
        self.endpoint_url = cfg.get("endpoint_url")
        self.aws_access_key_id = cfg.get("aws_access_key_id")
        self.aws_secret_access_key = cfg.get("aws_secret_access_key")
        self.verify = cfg.get("verify", True)

    def _client(self):
        try:
            import boto3
        except ImportError as e:
            msg = "s3 tools require 'boto3' (pip install teff[tools])"
            raise ImportError(msg) from e
        kwargs = {}
        if self.region:
            kwargs["region_name"] = self.region
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        kwargs["verify"] = self.verify
        return boto3.client("s3", **kwargs)

    def run(self, prefix: str = "", limit: int = 100) -> str:  # type: ignore[override]
        if not self.bucket:
            raise ValueError("s3_list requires 'bucket' in config")
        response = self._client().list_objects_v2(
            Bucket=self.bucket, Prefix=prefix, MaxKeys=limit
        )
        keys = [obj["Key"] for obj in response.get("Contents", [])]
        return "\n".join(keys) if keys else "no objects found"


class S3GetTool(S3Tool):
    """Download an object from an S3 bucket and return its contents."""

    name = "s3_get"
    description = "Download an object from an S3 bucket and return its contents"

    def run(self, key: str = "", max_chars: int = 50000) -> str:  # type: ignore[override]
        if not self.bucket:
            raise ValueError("s3_get requires 'bucket' in config")
        if not key:
            raise ValueError("key is required")
        body = self._client().get_object(Bucket=self.bucket, Key=key)["Body"]
        return body.read(max_chars).decode("utf-8", errors="replace")


class S3PutTool(S3Tool):
    """Upload text content to an object in an S3 bucket."""

    name = "s3_put"
    description = "Upload text content to an object in an S3 bucket"

    def run(self, key: str = "", content: str = "") -> str:  # type: ignore[override]
        if not self.bucket:
            raise ValueError("s3_put requires 'bucket' in config")
        if not key:
            raise ValueError("key is required")
        self._client().put_object(Bucket=self.bucket, Key=key, Body=content.encode())
        return f"uploaded {len(content)} bytes to {key}"
