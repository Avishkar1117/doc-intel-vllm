"""Client-IP extraction shared by every rate-limited route (one copy, so the trust rule
below can't drift between /extract and /demo)."""

from starlette.requests import Request


def client_ip(request: Request) -> str:
    """Caller address as seen by Azure Container Apps' ingress, spoof-resistant."""
    # Azure Container Apps' ingress is a reverse proxy - request.client.host can be the
    # proxy's own address rather than the real caller, which would put every caller in one
    # shared bucket. The ingress *appends* the real client IP to any X-Forwarded-For the
    # caller sent, and only that rightmost entry is trustworthy (Microsoft's ingress docs);
    # reading the leftmost let a caller dodge the limiter by rotating a fake header value
    # (reproduced live: 20x401 then 429 on one fake value, a fresh 401 on the next).
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"
