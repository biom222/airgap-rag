import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "offline_bundle.py"


def write_bundle(root: Path) -> None:
    artifacts = []
    files = {
        "deploy/docker-compose.yml": b"services: {}\n",
        "images/docker-images.tar": b"image archive",
        "sbom/airgap-rag.spdx.json": b"{}\n",
        "source/airgap-rag-source.tar.gz": b"source archive",
        "wheelhouse/airgap_rag-0.1.0-py3-none-any.whl": b"wheel",
    }
    roles = {
        "deploy": "deployment",
        "images": "docker-images",
        "sbom": "sbom",
        "source": "source",
        "wheelhouse": "wheelhouse",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
                "role": roles[relative.split("/", 1)[0]],
            }
        )
    manifest = {
        "schema_version": 1,
        "product": "AirGapRAG",
        "images": [],
        "models": [],
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    entries = [(item["sha256"], item["path"]) for item in artifacts]
    entries.append((hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "manifest.json"))
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in entries), encoding="utf-8"
    )


def validate(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "validate", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_offline_bundle_validator_accepts_complete_bundle(tmp_path: Path) -> None:
    write_bundle(tmp_path)

    result = validate(tmp_path)

    assert result.returncode == 0
    assert "Offline bundle is valid" in result.stdout


def test_offline_bundle_validator_rejects_tampered_artifact(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    (tmp_path / "images" / "docker-images.tar").write_bytes(b"tampered")

    result = validate(tmp_path)

    assert result.returncode == 1
    assert "mismatch" in result.stderr


def test_offline_compose_disables_builds_and_pulls() -> None:
    compose = (SCRIPT.parents[1] / "docker-compose.offline.yml").read_text(encoding="utf-8")

    assert "build: !reset null" in compose
    assert compose.count("pull_policy: never") == 10
    assert compose.count("internal: true") == 2
    assert "networks: [edge, frontend]" in compose


def test_offline_image_lock_has_no_floating_latest_tags() -> None:
    lock = json.loads((SCRIPT.parents[1] / "offline" / "images.lock.json").read_text())
    images = [lock["application_image"], *lock["images"].values()]

    assert all(":" in image for image in images)
    assert all(not image.endswith(":latest") for image in images)
