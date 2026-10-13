from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / "offline" / "images.lock.json"
MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "SHA256SUMS"
DEPLOYMENT_PATHS = (
    ".env.example",
    "docker-compose.yml",
    "docker-compose.production.yml",
    "docker-compose.offline.yml",
    "grafana",
    "nginx",
    "prometheus",
    "scripts/initialize-production.ps1",
    "scripts/validate-offline-bundle.ps1",
    "scripts/start-offline.ps1",
    "scripts/validate-offline-bundle.sh",
    "scripts/start-offline.sh",
)


class BundleError(RuntimeError):
    """Offline bundle cannot be created or validated safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    capture: bool = False,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise BundleError(f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return (result.stdout or "").strip()


def load_image_lock(path: Path = LOCK_FILE) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"Cannot read image lock {path}: {exc}") from exc
    if data.get("schema_version") != 1:
        raise BundleError("Unsupported image lock schema")
    images = data.get("images")
    app_image = data.get("application_image")
    if not isinstance(images, dict) or not images or not isinstance(app_image, str):
        raise BundleError("Image lock must define application_image and images")
    for service, image in {"application": app_image, **images}.items():
        if not isinstance(image, str) or ":" not in image or image.endswith(":latest"):
            raise BundleError(f"Image {service!r} is not pinned to an explicit tag: {image!r}")
    return data


def artifact_role(relative_path: str) -> str:
    if relative_path == "images/docker-images.tar":
        return "docker-images"
    if relative_path.startswith("wheelhouse/"):
        return "wheelhouse"
    if relative_path.startswith("models/"):
        return "model"
    if relative_path.startswith("sbom/"):
        return "sbom"
    if relative_path.startswith("source/"):
        return "source"
    return "deployment"


def iter_artifact_files(bundle: Path) -> Iterable[Path]:
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name not in {MANIFEST_NAME, CHECKSUMS_NAME}:
            yield path


def build_artifact_index(bundle: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(bundle).as_posix(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "role": artifact_role(path.relative_to(bundle).as_posix()),
        }
        for path in iter_artifact_files(bundle)
    ]


def write_checksums(bundle: Path, artifacts: list[dict[str, Any]]) -> None:
    manifest = bundle / MANIFEST_NAME
    entries = [(item["sha256"], item["path"]) for item in artifacts]
    entries.append((sha256_file(manifest), MANIFEST_NAME))
    payload = "".join(f"{digest}  {path}\n" for digest, path in entries)
    (bundle / CHECKSUMS_NAME).write_text(payload, encoding="utf-8", newline="\n")


def validate_bundle(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / MANIFEST_NAME
    checksums_path = bundle / CHECKSUMS_NAME
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise BundleError("Bundle must contain manifest.json and SHA256SUMS")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"Invalid manifest JSON: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise BundleError("Unsupported bundle manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise BundleError("Manifest artifacts must be a list")

    roles: set[str] = set()
    manifest_paths: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise BundleError("Malformed artifact entry")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or relative.startswith(("/", "\\"))
            or ".." in Path(relative).parts
        ):
            raise BundleError(f"Unsafe artifact path: {relative!r}")
        if relative in manifest_paths:
            raise BundleError(f"Duplicate artifact path: {relative}")
        manifest_paths.add(relative)
        path = bundle / relative
        if not path.is_file():
            raise BundleError(f"Missing artifact: {relative}")
        if path.stat().st_size != item.get("size"):
            raise BundleError(f"Artifact size mismatch: {relative}")
        if sha256_file(path) != item.get("sha256"):
            raise BundleError(f"Artifact checksum mismatch: {relative}")
        role = item.get("role")
        if isinstance(role, str):
            roles.add(role)

    required_roles = {"deployment", "docker-images", "sbom", "source", "wheelhouse"}
    missing_roles = required_roles - roles
    if missing_roles:
        raise BundleError(f"Bundle is missing required artifact roles: {sorted(missing_roles)}")

    expected_sums = {item["path"]: item["sha256"] for item in artifacts if isinstance(item, dict)}
    expected_sums[MANIFEST_NAME] = sha256_file(manifest_path)
    recorded_sums: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise BundleError(f"Malformed SHA256SUMS line: {line!r}") from exc
        recorded_sums[relative] = digest
    if recorded_sums != expected_sums:
        raise BundleError("SHA256SUMS does not match the manifest")
    return manifest


def copy_deployment(bundle: Path) -> None:
    deploy = bundle / "deploy"
    for relative in DEPLOYMENT_PATHS:
        source = ROOT / relative
        destination = deploy / relative
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def create_source_archive(destination: Path) -> None:
    tracked = run(["git", "ls-files", "-z"], capture=True)
    files = [entry for entry in tracked.split("\0") if entry]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for relative in files:
            source = ROOT / relative
            if source.is_file():
                archive.add(source, arcname=f"airgap-rag/{relative}")


def image_inspection(image: str) -> dict[str, Any]:
    payload = run(["docker", "image", "inspect", image], capture=True)
    inspection = json.loads(payload)[0]
    return {
        "reference": image,
        "image_id": inspection["Id"],
        "repo_digests": inspection.get("RepoDigests") or [],
        "architecture": inspection.get("Architecture"),
        "os": inspection.get("Os"),
    }


def create_sbom(destination: Path, images: list[dict[str, Any]], app_image: str) -> None:
    package_script = (
        "import importlib.metadata as m,json;"
        "print(json.dumps(sorted([{'name':d.metadata.get('Name',d.name),'version':d.version} "
        "for d in m.distributions()],key=lambda x:(x['name'].lower(),x['version']))))"
    )
    python_packages = json.loads(
        run(["docker", "run", "--rm", app_image, "python", "-c", package_script], capture=True)
    )
    packages: list[dict[str, Any]] = []
    relationships: list[dict[str, str]] = []
    for index, image in enumerate(images, start=1):
        spdx_id = f"SPDXRef-ContainerImage-{index}"
        packages.append(
            {
                "name": image["reference"],
                "SPDXID": spdx_id,
                "versionInfo": image["image_id"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:docker/{image['reference'].replace(':', '@')}",
                    }
                ],
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": spdx_id,
            }
        )
    for index, package in enumerate(python_packages, start=1):
        safe_name = "".join(
            character if character.isalnum() else "-" for character in package["name"]
        )
        spdx_id = f"SPDXRef-Python-{safe_name}-{index}"
        packages.append(
            {
                "name": package["name"],
                "SPDXID": spdx_id,
                "versionInfo": package["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{package['name']}@{package['version']}",
                    }
                ],
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-ContainerImage-1",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": spdx_id,
            }
        )
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "AirGapRAG offline bundle",
        "documentNamespace": f"https://airgap-rag.local/spdx/{uuid4()}",
        "creationInfo": {
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "creators": ["Tool: scripts/offline_bundle.py"],
        },
        "packages": packages,
        "relationships": relationships,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def create_wheelhouse(bundle: Path, app_image: str) -> None:
    wheelhouse = bundle / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "0",
            "--volume",
            f"{ROOT}:/src:ro",
            "--volume",
            f"{wheelhouse}:/wheelhouse",
            "--workdir",
            "/src",
            app_image,
            "sh",
            "-ec",
            (
                "mkdir -p /tmp/wheel-source/src; "
                "cp /src/pyproject.toml /src/README.md /tmp/wheel-source/; "
                "cp -R /src/src/airgap_rag /tmp/wheel-source/src/; "
                "cd /tmp/wheel-source; "
                "python -m pip wheel --wheel-dir /wheelhouse ."
            ),
        ]
    )
    if not any(wheelhouse.glob("*.whl")):
        raise BundleError("Wheelhouse creation produced no wheels")


def find_ollama_volume(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    output = run(
        ["docker", "volume", "ls", "--filter", "name=ollama_data", "--format", "{{.Name}}"],
        capture=True,
    )
    matches = [
        line.strip() for line in output.splitlines() if line.strip().endswith("_ollama_data")
    ]
    return matches[0] if len(matches) == 1 else None


def archive_ollama_volume(bundle: Path, app_image: str, volume: str) -> dict[str, str]:
    models = bundle / "models"
    models.mkdir(parents=True, exist_ok=True)
    destination = models / "ollama-models.tar"
    run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "0",
            "--volume",
            f"{volume}:/source:ro",
            "--volume",
            f"{models}:/bundle",
            app_image,
            "tar",
            "-cf",
            "/bundle/ollama-models.tar",
            "-C",
            "/source",
            ".",
        ]
    )
    if not destination.is_file() or destination.stat().st_size <= 1024:
        raise BundleError(f"Ollama volume {volume!r} is empty")
    return {"kind": "ollama-volume", "source": volume, "path": "models/ollama-models.tar"}


def archive_ollama_directory(bundle: Path, source: Path) -> dict[str, str]:
    if not (source / "manifests").is_dir() or not (source / "blobs").is_dir():
        raise BundleError(f"Not an Ollama models directory: {source}")
    destination = bundle / "models" / "ollama-models.tar"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as archive:
        archive.add(source, arcname="models")
    if destination.stat().st_size <= 1024:
        raise BundleError(f"Ollama models directory is empty: {source}")
    return {
        "kind": "ollama-model-store",
        "source": "host-ollama-model-store",
        "path": "models/ollama-models.tar",
    }


def write_bundle_environment(bundle: Path, lock: dict[str, Any], *, ollama_enabled: bool) -> None:
    names = {
        "AIRGAP_APP_IMAGE": lock["application_image"],
        "AIRGAP_MODE": "true",
        "LLM_PROVIDER": "ollama" if ollama_enabled else "mock",
        "OLLAMA_BASE_URL": "http://ollama:11434",
    }
    names.update(
        {f"AIRGAP_{service.upper()}_IMAGE": image for service, image in lock["images"].items()}
    )
    payload = "".join(f"{name}={value}\n" for name, value in names.items())
    (bundle / "deploy" / ".env.bundle").write_text(payload, encoding="utf-8", newline="\n")


def build_bundle(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise BundleError(f"Output already exists: {output}; use --force to replace it")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building-{uuid4().hex}"
    staging.mkdir()
    try:
        lock = load_image_lock(args.image_lock)
        app_image = lock["application_image"]
        if not args.skip_build:
            run(["docker", "build", "--tag", app_image, "."])
        external_images = list(lock["images"].values())
        if args.pull:
            for image in external_images:
                run(["docker", "pull", image])
        image_references = [app_image, *external_images]
        images = [image_inspection(image) for image in image_references]

        copy_deployment(staging)
        create_source_archive(staging / "source" / "airgap-rag-source.tar.gz")
        create_wheelhouse(staging, app_image)
        create_sbom(staging / "sbom" / "airgap-rag.spdx.json", images, app_image)

        image_archive = staging / "images" / "docker-images.tar"
        image_archive.parent.mkdir(parents=True)
        run(["docker", "save", "--output", str(image_archive), *image_references])

        models: list[dict[str, str]] = []
        default_model_dir = Path.home() / ".ollama" / "models"
        ollama_model_dir = args.ollama_model_dir
        if ollama_model_dir is None and default_model_dir.is_dir():
            ollama_model_dir = default_model_dir
        ollama_volume = find_ollama_volume(args.ollama_volume) if ollama_model_dir is None else None
        if ollama_model_dir is not None:
            models.append(archive_ollama_directory(staging, ollama_model_dir.resolve()))
        elif ollama_volume is not None:
            models.append(archive_ollama_volume(staging, app_image, ollama_volume))
        elif not args.allow_empty_models:
            raise BundleError(
                "No unique Ollama model volume found; pass --ollama-volume or explicitly "
                "acknowledge a mock-only bundle with --allow-empty-models"
            )
        for model_file in args.model_file:
            source = model_file.resolve()
            if not source.is_file():
                raise BundleError(f"Model file does not exist: {source}")
            destination = staging / "models" / "files" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            models.append(
                {
                    "kind": "model-file",
                    "source": str(source),
                    "path": destination.relative_to(staging).as_posix(),
                }
            )
        write_bundle_environment(staging, lock, ollama_enabled=bool(models))

        artifacts = build_artifact_index(staging)
        source_commit = run(["git", "rev-parse", "HEAD"], capture=True)
        dirty = bool(run(["git", "status", "--porcelain"], capture=True))
        manifest = {
            "schema_version": 1,
            "product": "AirGapRAG",
            "application_version": "0.1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": source_commit,
            "source_dirty": dirty,
            "application_image": app_image,
            "images": images,
            "models": models,
            "artifacts": artifacts,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_checksums(staging, artifacts)
        validate_bundle(staging)
        staging.rename(output)
        return output
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Build and validate AirGapRAG offline bundles"
    )
    subparsers = argument_parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Create a self-contained offline release bundle")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--image-lock", type=Path, default=LOCK_FILE)
    build.add_argument("--ollama-volume")
    build.add_argument("--ollama-model-dir", type=Path)
    build.add_argument("--model-file", type=Path, action="append", default=[])
    build.add_argument("--allow-empty-models", action="store_true")
    build.add_argument("--skip-build", action="store_true")
    build.add_argument("--pull", action="store_true")
    build.add_argument("--force", action="store_true")
    validate = subparsers.add_parser("validate", help="Verify every bundle artifact checksum")
    validate.add_argument("bundle", type=Path)
    return argument_parser


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "build":
            output = build_bundle(args)
            print(f"Offline bundle created and validated: {output}")
        else:
            manifest = validate_bundle(args.bundle.resolve())
            print(
                f"Offline bundle is valid: {len(manifest['artifacts'])} artifacts, "
                f"{len(manifest['images'])} images, {len(manifest['models'])} model archives"
            )
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
