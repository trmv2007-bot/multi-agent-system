"""Bind and request authentication guards for the HTTP servers.

Both ``src.mcp.server`` and ``src.a2a.server`` can serve over HTTP, and both can
reach far more than the network stack: ``team_task`` and ``research_report`` run
agents with tools, and with ``COMPUTER_USE=true`` those tools drive the desktop.
So a server that listens beyond this machine without a token is remote control.

The rule enforced here is one line long: listen on loopback by default, and if a
caller widens the bind, make it prove it set a token first, then require that
token on every request.
"""

import ipaddress
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

AUTH_ENV_VAR = "SERVER_AUTH_TOKEN"
LOOPBACK_HOST = "127.0.0.1"


def is_loopback(host: str) -> bool:
    """Whether binding to ``host`` stays on this machine."""
    name = host.strip().lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def require_authenticated_bind(
    host: str,
    token: str | None,
    *,
    server: str,
    host_env: str,
    token_env: str = AUTH_ENV_VAR,
) -> None:
    """Refuse to listen beyond this machine without a token configured.

    Raises ``SystemExit`` with the exact environment variable to set, so a
    mistyped command fails loudly instead of quietly exposing the agents.
    """
    if is_loopback(host) or token:
        return

    raise SystemExit(
        f"Refusing to start the {server} server on {host!r}: it is reachable from the\n"
        f"network and no token is configured.\n"
        f"  Either bind to loopback by unsetting {host_env} (default {LOOPBACK_HOST}),\n"
        f"  or set {token_env} to a shared secret and send it as\n"
        f"  'Authorization: Bearer <token>' on every request."
    )


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Require ``Authorization: Bearer <token>`` on every request when a token is set."""

    def __init__(self, app, token: str) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        presented = _bearer_token(request)
        if presented is None or not secrets.compare_digest(presented, self.token):
            return JSONResponse(
                {"error": "missing or invalid token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


def _bearer_token(request: Request) -> str | None:
    """The token from the Authorization header, or None when it is absent or malformed."""
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value.strip()


def add_token_middleware(app, token: str | None) -> None:  # type: ignore[no-untyped-def]
    """Install the bearer check when a token is configured, and leave the app alone otherwise."""
    if token:
        app.add_middleware(BearerTokenMiddleware, token=token)


def auth_headers(token: str | None) -> dict[str, str]:
    """Request headers for calling a token-protected server."""
    return {"Authorization": f"Bearer {token}"} if token else {}
