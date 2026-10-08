"""Integrity guard for the post-hoc addendum; never writes primary artifacts."""
import hashlib
import json

from src.common import digest, path, write_json

OUTPUT = "results/supplementary/robustfov_bet"
MASKS = "results/bet_robustfov"
METHOD = "bet_robustfov"
PROTOCOL_COMMIT = "b40aeca"


def frozen_state():
    before = json.loads(path(f"{OUTPUT}/frozen_before.json").read_text())
    current = {name: digest(path(name)) for name in before["files"]}
    readme = path("README.md").read_bytes()
    block = readme.split(b"<!-- RESULTS:START -->", 1)[1].split(b"<!-- RESULTS:END -->", 1)[0]
    block_hash = hashlib.sha256(block).hexdigest()
    changed = [name for name, value in current.items() if value != before["files"][name]]
    if block_hash != before["readme_results_block_sha256"]:
        changed.append("README RESULTS block")
    if changed:
        raise RuntimeError(f"Frozen primary artifacts changed: {changed}")
    return {"files": current, "readme_results_block_sha256": block_hash}


def verify_frozen():
    state = frozen_state()
    write_json(path(f"{OUTPUT}/frozen_after.json"), {**state, "all_unchanged": True})
    return state
