from fastapi import Header, HTTPException

from app.config import API_KEYS

# roles are simple and additive: reader can only search, admin can also
# ingest. good enough for a single-service demo — a real deployment would
# put this behind an identity provider instead of a static key map
ROLE_PERMISSIONS = {
    "reader": {"search"},
    "admin": {"search", "ingest"},
}


def require_permission(permission: str):
    def dependency(x_api_key: str = Header(...)) -> str:
        role = API_KEYS.get(x_api_key)
        if role is None:
            raise HTTPException(status_code=401, detail="invalid API key")
        if permission not in ROLE_PERMISSIONS.get(role, set()):
            raise HTTPException(status_code=403, detail=f"role '{role}' cannot '{permission}'")
        return x_api_key

    return dependency
