from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from airgap_rag.auth import Authenticator, Principal

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> Principal:
    authenticator = cast(Authenticator, request.app.state.authenticator)
    token = credentials.credentials if credentials is not None else None
    principal = authenticator.authenticate(token)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется действующий Bearer API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.principal = principal
    return principal


def require_admin(
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> Principal:
    if not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Для операции требуется роль admin.",
        )
    return principal
