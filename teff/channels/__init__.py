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
    router    = create_http_router(assistant)             # mount into an app
"""

from teff.channels.factory import build_assistant, build_webhook, load_channels
from teff.channels.reply import reply_from_state, reply_text, turn_response
from teff.channels.telegram import TelegramChannel
from teff.channels.webhook import WebhookChannel


def create_http_app(assistant, *, dependencies=None, turn_kwargs=None):
    """Build the HTTP/SSE FastAPI app for *assistant* (needs ``teff[channels]``).

    ``dependencies`` are FastAPI ``Depends`` objects applied to every
    non-health endpoint; ``turn_kwargs`` is a ``(owner, session_id) -> kwargs``
    factory merged into every ``Assistant.run``/``stream`` call.
    """
    from teff.channels.http import create_http_app as _factory

    return _factory(assistant, dependencies=dependencies, turn_kwargs=turn_kwargs)


def create_http_router(assistant, *, dependencies=None, turn_kwargs=None):
    """Build the HTTP/SSE routes for *assistant* as a mountable APIRouter.

    Use it to embed a channel into an existing app:

        from teff.channels import create_http_router
        app.include_router(create_http_router(assistant))
    """
    from teff.channels.http import create_http_router as _factory

    return _factory(assistant, dependencies=dependencies, turn_kwargs=turn_kwargs)


def HTTPChannel(assistant, *, dependencies=None, turn_kwargs=None):  # noqa: N802
    """Build an :class:`~teff.channels.http.HTTPChannel` (needs ``teff[channels]``)."""
    from teff.channels.http import HTTPChannel as _cls

    return _cls(assistant, dependencies=dependencies, turn_kwargs=turn_kwargs)


__all__ = [
    "build_assistant",
    "build_webhook",
    "load_channels",
    "reply_from_state",
    "reply_text",
    "turn_response",
    "create_http_app",
    "create_http_router",
    "HTTPChannel",
    "TelegramChannel",
    "WebhookChannel",
]
