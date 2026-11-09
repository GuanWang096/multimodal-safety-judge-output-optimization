"""Signer-ID alias probe in a single isolated node container.

No key material in argv/output. Strict mode is a candidate verifier guard,
not a change to the original protocol source or the live consensus service.
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
protocol.public_key = lambda node: Ed25519PublicKey.from_public_bytes(base64.b64decode(REGISTRY["public_keys"][str(node)]))
original_verify = protocol.verify_message


def strict_verify(message, allowed_members):
    # Canonical node IDs are JSON integers, not strings, bools or floats.
    if type(message.get("sender")) is not int:
        return False
    return original_verify(message, allowed_members)


def handle(request):
    action = request["action"]
    if action == "probe":
        return {"node": NODE, "visible_secret_files": sorted(p.name for p in Path("/run/node-secrets").iterdir()),
                "other_private_keys_readable": [n for n in range(4) if n != NODE and os.access(f"/run/node-secrets/node_{n}.key", os.R_OK)]}
    if action == "sign_with_own_key":
        message = request["message"].copy()
        message["signature"] = base64.b64encode(OWN.sign(protocol.canonical_bytes(message))).decode()
        return {"message": message, "actual_signing_node": NODE}
    if action == "certificate":
        protocol.verify_message = strict_verify if request["mode"] == "candidate_strict_integer" else original_verify
        messages = request["messages"]
        expected = request["expected"]
        individual = [protocol.verify_message(m, expected["allowed_members"]) for m in messages]
        try:
            accepted = protocol.validate_certificate(messages, **expected)
            return {"accepted": True, "raw_signers": [m["sender"] for m in accepted],
                    "canonical_verified_ids": sorted({int(m["sender"]) for m in accepted}), "individual_verified": individual}
        except ValueError as exc:
            return {"accepted": False, "reason": str(exc), "individual_verified": individual}
    raise ValueError("unknown action")


if __name__ == "__main__":
    try:
        print(json.dumps(handle(json.loads(base64.b64decode(sys.argv[1])))))
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}))
