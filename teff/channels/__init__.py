"""Channel adapters: run one ``workflow.yaml`` over many transports.

Constitution Principle IX: observability and a single source of truth.
The channel layer keeps the workflow YAML as the one executable spec and
binds transport adapters (HTTP/SSE, Telegram, generic webhooks) onto the
same durable :class:`~teff.assistant.Assistant` service, so interrupt
handling, checkpoints and message history behave identically on every
surface.

Importing this package is dependency-free (stdlib + httpx): the HTTP
adapter needs the optional ``teff[channels]`` extra and is imported
lazily via :func:`create_http_app`.

Public API::

    assistant = build_assistant("workflow.yaml")          # one durable service
    hook      = build_webhook(assistant, spec)            # generic webhook
    bot       = TelegramChannel(assistant, token=...)     # polling/webhook
    app       = create_http_app(assistant)                # FastAPI + SSE
"""

from teff.channels.factory import build_assistant, build_webhook, load_channels
from teff.channels.reply import reply_from_state, reply_text, turn_response
from teff.channels.telegram import TelegramChannel
from teff.channels.webhook import WebhookChannel


def create_http_app(assistant):
    """Build the HTTP/SSE FastAPI app for *assistant* (needs ``teff[channels]``)."""
    from teff.channels.http import create_http_app as _factory

    return _factory(assistant)


def HTTPChannel(assistant):  # noqa: N802
    """Build an :class:`~teff.channels.http.HTTPChannel` (needs ``teff[channels]``)."""
    from teff.channels.http import HTTPChannel as _cls

    return _cls(assistant)


__all__ = [
    "build_assistant",
    "build_webhook",
    "load_channels",
    "reply_from_state",
    "reply_text",
    "turn_response",
    "create_http_app",
    "HTTPChannel",
    "TelegramChannel",
    "WebhookChannel",
]
