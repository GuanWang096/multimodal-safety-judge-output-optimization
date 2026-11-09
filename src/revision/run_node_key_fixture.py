"""Four-container, node-local-key authentication fixture. Run in Ubuntu WSL.

No ledger calls or model inference. Original protocol.py is read-only. Private
keys exist only in a permission-restricted temporary host directory and are
removed on exit; registry, signed messages and observations are retained.
"""
import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "results/reproduced/revision/protocol/node-key-fixture-20261031-v3"
IMAGE_TAG = "zte-rggpbft:revision-r1-20261030"
LABEL = "zte.revision.fixture=node-key-20261031-v3"


def command(arguments, **kwargs):
    return subprocess.run(arguments, text=True, capture_output=True, check=True, timeout=60, **kwargs).stdout.strip()


def image_id():
    return command(["docker", "image", "inspect", IMAGE_TAG, "--format", "{{.Id}}"])


def docker_bind_path(path):
    # This WSL installation invokes the Windows Docker CLI/npipe context.
    # Convert both /mnt/c paths and the private /tmp directory (WSL UNC).
    return command(["wslpath", "-w", str(Path(path).resolve())])


def preflight():
    image = image_id()
    witness = "from pathlib import Path; from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey; import cryptography,base64,json,sys; assert json.loads(base64.b64decode(sys.argv[1]))=={'action':'probe'}; assert Path('/probe/marker').read_text()=='public-preflight-marker'; assert 'SIGNED_FIELDS' in Path('/probe_protocol.py').read_text(); k=Ed25519PrivateKey.from_private_bytes(bytes(range(32))); m=b'revision-fixed-preflight'; k.public_key().verify(k.sign(m),m); print('KEY_FIXTURE_MOUNT_ARGV_WITNESS_PASS '+cryptography.__version__)"
    with tempfile.TemporaryDirectory(prefix="zte-revision-key-probe-") as probe:
        probe_dir = Path(probe); probe_dir.chmod(0o700)
        marker = probe_dir / "marker"; marker.write_text("public-preflight-marker"); marker.chmod(0o600)
        observed = command(["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
                            "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
                            "--mount", f"type=bind,source={docker_bind_path(probe_dir)},target=/probe,readonly",
                            "--mount", f"type=bind,source={docker_bind_path(ROOT / 'src/rggpbft/protocol.py')},target=/probe_protocol.py,readonly",
                            "--entrypoint", "python", image, "-B", "-c", witness,
                            base64.b64encode(json.dumps({"action":"probe"}).encode()).decode()])
    assert observed == "KEY_FIXTURE_MOUNT_ARGV_WITNESS_PASS 44.0.1"
    print(json.dumps({"status": "preflight_pass", "image_id": image, "witness": observed, "gpu_count": 0}), flush=True)
    return image


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    image = preflight()
    if args.preflight: return
    if OUT.exists(): raise RuntimeError("Fixture output already exists; do not overwrite or silently rerun")
    OUT.mkdir()
    containers = []
    observations = []
    transcript = []
    status = {"state": "running", "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "image_id": image, "container_count": 4, "network_mode": "none", "node_count": 4,
              "scope": "isolated_container_protocol_authentication_fixture_not_live_consensus_or_independent_organizations",
              "trust": "host_orchestrator_and_Docker_trusted; node_containers_untrusted_with_own_key_only",
              "key_lifecycle": "random_once_per_run; no_private_keys_in_retained_artifacts", "gpu_count": 0,
              "checks": observations}
    def save():
        (OUT / "results.json").write_text(json.dumps(status, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    save()
    def call(node, request):
        encoded_request = base64.b64encode(json.dumps(request).encode()).decode()
        response = json.loads(command(["docker", "exec", containers[node], "python", "-B", "/fixture/key_fixture_worker.py", encoded_request]))
        transcript.append({"node": node, "request": request, "response": response})
        with (OUT / "transcript.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(transcript[-1])+"\n")
        return response
    def check(name, condition, details=None):
        observations.append({"name": name, "passed": bool(condition), "details": details})
        save()
        if not condition: raise RuntimeError("Authentication fixture failed: " + name)
    try:
        with tempfile.TemporaryDirectory(prefix="zte-revision-node-keys-") as temp:
            secret_root = Path(temp); os.chmod(secret_root, 0o700)
            public = {}
            for node in range(4):
                directory = secret_root / str(node); directory.mkdir(mode=0o700)
                key = Ed25519PrivateKey.generate()
                keyfile = directory / f"node_{node}.key"
                keyfile.write_bytes(key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()))
                keyfile.chmod(0o600)
                public[str(node)] = base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
            registry = {"schema": "revision-public-registry-v1", "membership_epoch": 0, "public_keys": public}
            (OUT / "public_registry.json").write_text(json.dumps(registry, indent=2)+"\n", encoding="utf-8")
            for node in range(4):
                name = f"zte-revision-key-20261031-v3-{node}"
                # A bind of one node-specific directory, not the directory of all keys.
                mounts = [(secret_root / str(node), "/run/node-secrets"),
                          (ROOT / "src/rggpbft", "/fixture/protocol"),
                          (CODE / "key_fixture_worker.py", "/fixture/key_fixture_worker.py"),
                          (OUT / "public_registry.json", "/fixture/public_registry.json")]
                cli = ["docker", "run", "-d", "--name", name, "--label", LABEL, "--network", "none", "--read-only",
                       "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
                       "-e", f"FIXTURE_NODE_ID={node}", "--entrypoint", "python"]
                for source, target in mounts:
                    cli += ["--mount", f"type=bind,source={docker_bind_path(source)},target={target},readonly"]
                cli += [image, "-B", "-c", "import time; time.sleep(3600)"]
                cid = command(cli); containers.append(cid)
                details = json.loads(command(["docker", "inspect", cid]))[0]
                safe_inspect = {"id": cid, "name": name, "image": details["Image"], "user": details["Config"]["User"],
                                "network": details["HostConfig"]["NetworkMode"], "readonly_rootfs": details["HostConfig"]["ReadonlyRootfs"],
                                "cap_drop": details["HostConfig"]["CapDrop"], "security_opt": details["HostConfig"]["SecurityOpt"],
                                "mounts": [{"destination": m["Destination"], "rw": m["RW"], "type": m["Type"]} for m in details["Mounts"]]}
                (OUT / f"container_{node}.json").write_text(json.dumps(safe_inspect, indent=2)+"\n", encoding="utf-8")
                check(f"container_{node}_mounts", details["Image"] == image and not any(m["RW"] for m in details["Mounts"]) and len(details["Mounts"]) == 4, safe_inspect)
                probe = call(node, {"action": "probe"})
                check(f"node_{node}_secret_visibility", probe["visible_secret_files"] == [f"node_{node}.key"] and not probe["other_node_private_keys_readable"] and not probe["docker_socket_visible"], probe)
            digest = hashlib.sha256(b"synthetic-authentication-fixture-not-model-evidence").hexdigest()
            def unsigned(sender): return {"type": "COMMIT", "sender": sender, "view": 0, "sequence": 7, "digest": digest, "group": 0, "payload": {"epoch": 0}}
            signed = [call(n, {"action": "sign", "message": unsigned(n)})["message"] for n in range(4)]
            for verifier in range(4):
                for sender in range(4):
                    check(f"valid_sender_{sender}_verifier_{verifier}", call(verifier, {"action": "verify", "message": signed[sender], "allowed_members": list(range(4))})["accepted"])
            denied = call(0, {"action": "sign", "message": unsigned(1)})
            check("honest_signing_api_rejects_other_identity", denied.get("error_type") == "PermissionError", denied)
            forged = call(0, {"action": "forge_with_own_key", "message": unsigned(1)})["message"]
            check("malicious_own_key_cannot_impersonate_node1", not call(3, {"action": "verify", "message": forged, "allowed_members": list(range(4))})["accepted"])
            legacy = call(0, {"action": "legacy_signature", "message": unsigned(1)})["message"]
            check("public_legacy_key_rejected", not call(3, {"action": "verify", "message": legacy, "allowed_members": list(range(4))})["accepted"])
            tampered = copy.deepcopy(signed[0]); tampered["digest"] = "changed"
            check("changed_digest_rejected", not call(3, {"action": "verify", "message": tampered, "allowed_members": list(range(4))})["accepted"])
            check("unregistered_sender_rejected", not call(3, {"action": "verify", "message": signed[0], "allowed_members": [1,2,3]})["accepted"])
            expected = {"message_type": "COMMIT", "view": 0, "sequence": 7, "digest": digest, "group": 0, "allowed_members": list(range(4)), "threshold": 3}
            certificate = call(3, {"action": "certificate", "messages": signed[:3], "expected": expected})
            check("valid_three_signer_certificate", certificate.get("accepted") and certificate.get("signers") == [0,1,2], certificate)
            duplicate = call(3, {"action": "certificate", "messages": [signed[0],signed[0],signed[1]], "expected": expected})
            check("duplicate_signer_certificate_rejected", duplicate.get("accepted") is False, duplicate)
            stale = call(3, {"action": "certificate", "messages": signed[:3], "expected": {**expected, "view": 1}})
            check("stale_view_certificate_rejected", stale.get("accepted") is False, stale)
            status["state"] = "completed_pending_independent_audit"
            status["check_count"] = len(observations)
            status["sources"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(),CODE/"key_fixture_worker.py",ROOT/"src/rggpbft/protocol.py")}
            save()
    except Exception as exc:
        status["state"] = "failed"; status["error"] = repr(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            status["command_stdout"] = exc.stdout
            status["command_stderr"] = exc.stderr
        save(); raise
    finally:
        cleanup = []
        for cid in containers:
            label = command(["docker", "inspect", cid, "--format", '{{index .Config.Labels "zte.revision.fixture"}}'])
            if label != "node-key-20261031-v3": raise RuntimeError("Refusing to remove container with unexpected fixture label")
            command(["docker", "rm", "-f", cid]); cleanup.append(cid)
        status["removed_fixture_container_ids"] = cleanup
        status["ephemeral_private_directory_removed"] = "secret_root" in locals() and not secret_root.exists()
        status["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    print(json.dumps({"state": status["state"], "checks": len(observations), "private_keys_retained": False}), flush=True)


if __name__ == "__main__": main()
