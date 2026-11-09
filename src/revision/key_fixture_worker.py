"""One-shot authentication operations inside one node's isolated container.

Uses original protocol serialization/verification with injected public registry
and one private-key provider. This is not the live node/view-change state machine.
"""
import base64
import json
import os
from pathlib import Path
import sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
sys.dont_write_bytecode = True
sys.path.insert(0, "/fixture/protocol")
import protocol

NODE = int(os.environ["FIXTURE_NODE_ID"])
REGISTRY = json.loads(Path("/fixture/public_registry.json").read_text())
OWN = Ed25519PrivateKey.from_private_bytes(Path(f"/run/node-secrets/node_{NODE}.key").read_bytes())
legacy_private_key = protocol.private_key


def private(node):
    if node != NODE: raise PermissionError("node cannot access another node private key")
    return OWN


def public(node):
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(REGISTRY["public_keys"][str(node)]))


protocol.private_key = private
protocol.public_key = public


def handle(request):
    action = request["action"]
    if action == "probe":
        return {"node": NODE, "visible_secret_files": sorted(p.name for p in Path("/run/node-secrets").iterdir()),
                "other_node_private_keys_readable": [n for n in range(4) if n != NODE and os.access(f"/run/node-secrets/node_{n}.key", os.R_OK)],
                "docker_socket_visible": Path("/var/run/docker.sock").exists()}
    if action == "sign":
        m = request["message"]
        return {"message": protocol.make_message(m["type"], m["sender"], m["view"], m["sequence"], m["digest"], m["group"], m["payload"])}
    if action == "forge_with_own_key":
        # An attacker can lie about sender, but it still only has its own key.
        m = request["message"].copy()
        m["signature"] = base64.b64encode(OWN.sign(protocol.canonical_bytes(m))).decode()
        return {"message": m}
    if action == "legacy_signature":
        m = request["message"].copy()
        m["signature"] = base64.b64encode(legacy_private_key(m["sender"]).sign(protocol.canonical_bytes(m))).decode()
        return {"message": m}
    if action == "verify":
        return {"accepted": protocol.verify_message(request["message"], request["allowed_members"])}
    if action == "certificate":
        try:
            result = protocol.validate_certificate(request["messages"], **request["expected"])
            return {"accepted": True, "signers": [m["sender"] for m in result]}
        except ValueError as exc:
            return {"accepted": False, "reason": str(exc)}
    raise ValueError("unknown action")


if __name__ == "__main__":
    try:
        # Requests contain only public messages/operations. This machine's
        # Docker wrapper intentionally redirects stdin, so use encoded argv.
        result = handle(json.loads(base64.b64decode(sys.argv[1])))
        print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}))
