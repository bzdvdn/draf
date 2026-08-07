"""Session endpoints — inspect and delete durable runs.

All lookups are scoped to the authenticated caller (``X-User-Id`` via
:func:`~src.api.auth.router.require_user_id`): a session is only reachable
under its own owner namespace, so one caller cannot read or delete another
caller's sessions.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from src.api.auth.router import require_api_key, require_user_id

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/{chat_id}")
async def get_run(
    chat_id: str,
    request: Request,
    owner: str = Depends(require_user_id),
) -> dict:
    saved = await request.app.state.assistant.checkpointer.load(chat_id, owner=owner)
    if saved is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "chat_id": chat_id,
        "owner": owner,
        "next_node_id": saved.next_node_id,
        "iteration": saved.iteration,
        "state": saved.state,
    }


@router.delete("/{chat_id}")
async def delete_run(
    chat_id: str,
    request: Request,
    owner: str = Depends(require_user_id),
) -> dict:
    await request.app.state.assistant.checkpointer.delete(chat_id, owner=owner)
    return {"chat_id": chat_id, "status": "deleted"}
