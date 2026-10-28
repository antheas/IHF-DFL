import os
import platform
import subprocess
from solcx import install_solc, set_solc_version, compile_standard, get_installed_solc_versions
from pathlib import Path
import json
from utils.paths import repo_root

print(get_installed_solc_versions())

SOLC_VERSION = "0.8.9"
IS_ARM = platform.machine() in ("aarch64", "arm64")
root = repo_root(Path(__file__))

compile_kwargs = {}
if IS_ARM:
    # py-solc-x has no native ARM release for this old compiler. A caller may
    # still provide a native binary, otherwise we use the pinned npm module.
    solc_binary = os.environ.get("SOLC_BINARY")
    if solc_binary:
        compile_kwargs["solc_binary"] = solc_binary

use_solc_js = IS_ARM and not compile_kwargs and (root / "node_modules" / "solc").is_dir()

if not compile_kwargs and not use_solc_js:
    install_solc(SOLC_VERSION)
    set_solc_version(SOLC_VERSION)

# 2) Load sources
contracts_dir = root / "contracts"
harnesses_contracts_dir = root / "contracts" / "harnesses"
sources = {
    "OpenFLManager.sol": {"content": (contracts_dir / "OpenFLManager.sol").read_text(encoding="utf-8")},
    "OpenFLManager_NobodyIsKicked.sol": {"content": (harnesses_contracts_dir / "OpenFLManager_NobodyIsKicked.sol").read_text(encoding="utf-8")},
    "OpenFLModel.sol":   {"content": (contracts_dir / "OpenFLModel.sol").read_text(encoding="utf-8")},
    "OpenFLModel_NobodyIsKicked.sol": {"content": (harnesses_contracts_dir / "OpenFLModel_NobodyIsKicked.sol").read_text(encoding="utf-8")},
}

# 3) Compile
compiler_input = {
    "language": "Solidity",
    "sources": sources,
    "settings": {
        "optimizer": {"enabled": True, "runs": 200},
        "outputSelection": {"*": {"*": ["abi","evm.bytecode.object"]}}
    }
}

if use_solc_js:
    result = subprocess.run(
        ["node", str(root / "scripts" / "compile_solidity.js")],
        input=json.dumps(compiler_input),
        text=True,
        capture_output=True,
        cwd=root,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"solcjs failed:\n{result.stderr}")
    compiled = json.loads(result.stdout)
else:
    compiled = compile_standard(compiler_input, **compile_kwargs)

bytecode = compiled["contracts"]["OpenFLModel.sol"]["OpenFLModel"]["evm"]["bytecode"]["object"]

size_bytes = len(bytecode) // 2  # Each 2 hex chars = 1 byte
size_kb = size_bytes / 1024

print(f"Contract size OpenFLModel: {size_bytes} bytes ({size_kb:.2f} KB)")


bytecode = compiled["contracts"]["OpenFLModel_NobodyIsKicked.sol"]["OpenFLModel_NobodyIsKicked"]["evm"]["bytecode"]["object"]

size_bytes = len(bytecode) // 2  # Each 2 hex chars = 1 byte
size_kb = size_bytes / 1024

print(f"Contract size OpenFLModel_NobodyIsKicked: {size_bytes} bytes ({size_kb:.2f} KB)")


bytecode = compiled["contracts"]["OpenFLManager.sol"]["OpenFLManager"]["evm"]["bytecode"]["object"]

size_bytes = len(bytecode) // 2  # Each 2 hex chars = 1 byte
size_kb = size_bytes / 1024

print(f"Contract size OpenFLManager: {size_bytes} bytes ({size_kb:.2f} KB)")


bytecode = compiled["contracts"]["OpenFLManager_NobodyIsKicked.sol"]["OpenFLManager_NobodyIsKicked"]["evm"]["bytecode"]["object"]

size_bytes = len(bytecode) // 2  # Each 2 hex chars = 1 byte
size_kb = size_bytes / 1024

print(f"Contract size OpenFLManager_NobodyIsKicked: {size_bytes} bytes ({size_kb:.2f} KB)")


# 4) Extract artifacts
mgr = compiled["contracts"]["OpenFLManager.sol"]["OpenFLManager"]
mgr_nobody = compiled["contracts"]["OpenFLManager_NobodyIsKicked.sol"]["OpenFLManager_NobodyIsKicked"]
mdl = compiled["contracts"]["OpenFLModel.sol"]["OpenFLModel"]
mdl_nobody = compiled["contracts"]["OpenFLModel_NobodyIsKicked.sol"]["OpenFLModel_NobodyIsKicked"]

build = root / "artifacts" / "bytecode"
build.mkdir(parents=True, exist_ok=True)

# IMPORTANT: abi.txt should be JSON, because Python should json.load it later
(Path(build / "abi_mgr.txt")).write_text(json.dumps(mgr["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode_mgr.txt")).write_text(mgr["evm"]["bytecode"]["object"], encoding="utf-8")
(Path(build / "abi_mgr_nobody.txt")).write_text(json.dumps(mgr_nobody["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode_mgr_nobody.txt")).write_text(mgr_nobody["evm"]["bytecode"]["object"], encoding="utf-8")

(Path(build / "abi_model.txt")).write_text(json.dumps(mdl["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "abi_model_nobody.txt")).write_text(json.dumps(mdl_nobody["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode_model.txt")).write_text(mdl["evm"]["bytecode"]["object"], encoding="utf-8")
(Path(build / "bytecode_model_nobody.txt")).write_text(mdl_nobody["evm"]["bytecode"]["object"], encoding="utf-8")


print("Artifacts written to build/: abi.txt, OPEN_FL_MODEL_ABI.py, bytecode.txt, abi_model.txt, bytecode_model.txt and similar for nobody_is_kicked")
