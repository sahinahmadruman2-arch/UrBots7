from __future__ import annotations

from typing import Optional

from fastapi import Cookie, Header, HTTPException

from karllo.security import valid_session


def owner_required(
    karllo_session: Optional[str] = Cookie(default=None),
    authorization: Optional[str] = Header(default=None),
) -> str:
    token = karllo_session
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not valid_session(token):
        raise HTTPException(status_code=401, detail="Owner authentication required")
    return token
