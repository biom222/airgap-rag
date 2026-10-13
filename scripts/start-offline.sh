#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: $0 BUNDLE_DIRECTORY [PROJECT_NAME] [--with-ollama] [--with-observability]" >&2
    exit 2
fi

bundle=$(CDPATH= cd -- "$1" && pwd)
project_name=${2:-airgap-rag}
if [ "$#" -ge 2 ]; then
    shift 2
else
    shift 1
fi
profiles=""
with_ollama=false
for option in "$@"; do
    case "$option" in
        --with-ollama)
            with_ollama=true
            profiles="ollama${profiles:+,$profiles}"
            ;;
        --with-observability)
            profiles="observability${profiles:+,$profiles}"
            ;;
        *)
            echo "unknown option: $option" >&2
            exit 2
            ;;
    esac
done
deploy="$bundle/deploy"
"$deploy/scripts/validate-offline-bundle.sh" "$bundle"
docker load --input "$bundle/images/docker-images.tar"

if [ "$with_ollama" = true ]; then
    test -f "$bundle/models/ollama-models.tar" || {
        echo "The bundle has no Ollama model archive" >&2
        exit 1
    }
    app_image=$(sed -n 's/^AIRGAP_APP_IMAGE=//p' "$deploy/.env.bundle")
    volume="${project_name}_ollama_data"
    docker volume create "$volume" >/dev/null
    docker run --rm --user 0 \
        --volume "$volume:/restore" \
        --volume "$bundle/models:/bundle:ro" \
        "$app_image" \
        sh -ec 'test -z "$(ls -A /restore)" || { echo "Ollama volume is not empty" >&2; exit 20; }; tar -xf /bundle/ollama-models.tar -C /restore'
fi

COMPOSE_PROFILES="$profiles" docker compose \
    --project-name "$project_name" \
    --env-file "$deploy/.env.bundle" \
    --file "$deploy/docker-compose.yml" \
    --file "$deploy/docker-compose.production.yml" \
    --file "$deploy/docker-compose.offline.yml" \
    up --detach --pull never --wait

echo "AirGapRAG started without registry access."
