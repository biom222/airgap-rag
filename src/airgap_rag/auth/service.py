import hashlib
import hmac
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    ADMIN = "admin"
    USER = "user"


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    role: Role

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN


@dataclass(frozen=True, slots=True)
class _Credential:
    token_sha256: str
    principal: Principal


class AuthConfigurationError(ValueError):
    """Authentication settings are invalid and the service must not start."""


class Authenticator:
    def __init__(self, *, required: bool, principals_json: str) -> None:
        self.required = required
        self._credentials = self._parse(principals_json)
        if required and not self._credentials:
            raise AuthConfigurationError(
                "AUTH_REQUIRED=true requires at least one entry in AUTH_PRINCIPALS_JSON"
            )

    @staticmethod
    def token_digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def authenticate(self, token: str | None) -> Principal | None:
        if not self.required:
            return Principal(subject="local-development", role=Role.ADMIN)
        if token is None:
            return None
        candidate = self.token_digest(token)
        for credential in self._credentials:
            if hmac.compare_digest(candidate, credential.token_sha256):
                return credential.principal
        return None

    @staticmethod
    def _parse(raw: str) -> tuple[_Credential, ...]:
        try:
            records: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuthConfigurationError("AUTH_PRINCIPALS_JSON must be valid JSON") from exc
        if not isinstance(records, list):
            raise AuthConfigurationError("AUTH_PRINCIPALS_JSON must be a JSON array")

        credentials: list[_Credential] = []
        subjects: set[str] = set()
        digests: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise AuthConfigurationError("Each authentication principal must be an object")
            subject = record.get("subject")
            role_value = record.get("role")
            digest = record.get("token_sha256")
            if not isinstance(subject, str) or not subject.strip() or len(subject) > 100:
                raise AuthConfigurationError("Principal subject must contain 1-100 characters")
            if not isinstance(role_value, str):
                raise AuthConfigurationError("Principal role must be admin or user")
            try:
                role = Role(role_value)
            except (TypeError, ValueError) as exc:
                raise AuthConfigurationError("Principal role must be admin or user") from exc
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in digest)
            ):
                raise AuthConfigurationError("token_sha256 must be a 64-character hex digest")
            normalized_subject = subject.strip()
            normalized_digest = digest.lower()
            if normalized_subject in subjects or normalized_digest in digests:
                raise AuthConfigurationError("Principal subjects and token hashes must be unique")
            subjects.add(normalized_subject)
            digests.add(normalized_digest)
            credentials.append(
                _Credential(
                    token_sha256=normalized_digest,
                    principal=Principal(subject=normalized_subject, role=role),
                )
            )
        return tuple(credentials)
