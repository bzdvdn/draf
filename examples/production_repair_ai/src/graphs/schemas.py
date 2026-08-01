"""JSON schemas for structured LLM output in the repair workflow."""

from __future__ import annotations

#: Fields the extractor may pull from the conversation.  Everything is
#: optional: the extractor prompt instructs the model to return ``null``
#: for anything it cannot find.
PROJECT_INFO_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "room_type": {"type": "string"},
        "area": {"type": "number"},
        "ceiling_height": {"type": "number"},
        "budget": {"type": "number"},
        "style": {"type": "string"},
        "walls_area": {"type": "number"},
        "floor_area": {"type": "number"},
        "ceiling_area": {"type": "number"},
    },
}
