#!/bin/sh
set -eu

read_secret() {
    secret_path="$1"
    if [ ! -r "$secret_path" ]; then
        echo "Required secret is not readable: $secret_path" >&2
        exit 1
    fi
    tr -d '\r\n' < "$secret_path"
}

if [ "${APP_ENV:-}" = "production" ]; then
    database_password="$(read_secret /run/secrets/postgres_password)"
    redis_password="$(read_secret /run/secrets/redis_password)"
    export DATABASE_URL="postgresql+asyncpg://airgap:${database_password}@postgres:5432/airgap_rag"
    export REDIS_URL="redis://:${redis_password}@redis:6379/0"
    export QDRANT_API_KEY="$(read_secret /run/secrets/qdrant_api_key)"
    export AUTH_PRINCIPALS_JSON="$(cat /run/secrets/auth_principals_json)"
fi

exec "$@"
