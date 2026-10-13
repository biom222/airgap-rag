#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 BUNDLE_DIRECTORY" >&2
    exit 2
fi

bundle=$(CDPATH= cd -- "$1" && pwd)
test -f "$bundle/manifest.json"
test -f "$bundle/SHA256SUMS"
(cd "$bundle" && sha256sum --check --strict SHA256SUMS)
echo "Offline bundle checksums are valid."
