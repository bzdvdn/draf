"""Notification tools — send email via SMTP and Telegram bot messages."""

from teff.tool.tool import Tool


class SendEmailTool(Tool):
    """Send an email message via SMTP.

    Uses the stdlib ``smtplib``/``email`` modules — no extra dependency.

    Args:
        config: Optional dict with ``host`` (SMTP server), ``port``
            (default 587), ``username``, ``password``, ``from_addr``
            (sender address), ``starttls`` (default True).
    """

    name = "send_email"
    description = "Send an email message via SMTP"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.host = cfg.get("host", "")
        self.port = cfg.get("port", 587)
        self.username = cfg.get("username", "")
        self.password = cfg.get("password", "")
        self.from_addr = cfg.get("from_addr", "")
        self.starttls = cfg.get("starttls", True)

    def run(self, to: str = "", subject: str = "", body: str = "") -> str:  # type: ignore[override]
        if not self.host:
            raise ValueError("send_email requires 'host' in config")
        if not self.from_addr:
            raise ValueError("send_email requires 'from_addr' in config")
        if not to:
            raise ValueError("to is required")

        import smtplib
        from email.mime.text import MIMEText

        message = MIMEText(body, "plain", "utf-8")
        message["Subject"] = subject
        message["From"] = self.from_addr
        message["To"] = to

        with smtplib.SMTP(self.host, self.port, timeout=30) as server:
            if self.starttls:
                server.starttls()
            if self.username:
                server.login(self.username, self.password)
            server.sendmail(self.from_addr, to, message.as_string())
        return f"email sent to {to}"


class SendTelegramTool(Tool):
    """Send a message via the Telegram Bot API.

    Uses ``httpx`` (a core dependency). Config keys ``token`` (bot token)
    and ``chat_id`` (default recipient).

    Args:
        config: Optional dict with ``token`` and ``chat_id``.
    """

    name = "send_telegram"
    description = "Send a message via a Telegram bot"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.token = cfg.get("token", "")
        self.chat_id = cfg.get("chat_id", "")

    async def arun(self, text: str = "", chat_id: str = "") -> str:  # type: ignore[override]
        if not self.token:
            raise ValueError("send_telegram requires 'token' in config")
        target = chat_id or self.chat_id
        if not target:
            raise ValueError("chat_id is required")
        if not text:
            raise ValueError("text is required")

        import httpx

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, json={"chat_id": target, "text": text})
            response.raise_for_status()
        return f"telegram message sent to {target}"
