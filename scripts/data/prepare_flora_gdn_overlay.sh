#!/usr/bin/env bash
# Build a new isolated overlay only. Never install/replace the base Torch/Triton.
set -euo pipefail

usage() {
  echo 'Usage: prepare_flora_gdn_overlay.sh --root NEW_ABSOLUTE_DIR [--python /opt/flora/bin/python]'
}
GDN_ROOT=''
GDN_PYTHON='/opt/flora/bin/python'
while (($#)); do
  case "$1" in
    --root) GDN_ROOT="$2"; shift 2 ;;
    --python) GDN_PYTHON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ "$GDN_ROOT" != /* || "$GDN_ROOT" == / || -e "$GDN_ROOT" ]]; then
  echo 'Use a NEW absolute root. Existing paths are preserved.' >&2
  exit 2
fi
command -v "$GDN_PYTHON" >/dev/null
command -v g++ >/dev/null
command -v nvcc >/dev/null
mkdir -p "$(dirname "$GDN_ROOT")"
mkdir "$GDN_ROOT"
mkdir "$GDN_ROOT/wheels" "$GDN_ROOT/build-wheels" "$GDN_ROOT/build-tools" "$GDN_ROOT/site" "$GDN_ROOT/logs"
mkdir "$GDN_ROOT/pip-cache" "$GDN_ROOT/torch-extensions" "$GDN_ROOT/tmp" "$GDN_ROOT/triton-cache"
export PIP_CACHE_DIR="$GDN_ROOT/pip-cache"
export TORCH_EXTENSIONS_DIR="$GDN_ROOT/torch-extensions"
export TMPDIR="$GDN_ROOT/tmp"
export TRITON_CACHE_DIR="$GDN_ROOT/triton-cache"
exec > >(tee "$GDN_ROOT/logs/build.log") 2>&1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MAX_JOBS=2
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
# Official setup.py supplies explicit sm110 gencode. Do not ask cpp_extension
# to parse a possibly unsupported TORCH_CUDA_ARCH_LIST="11.0" string.
unset TORCH_CUDA_ARCH_LIST CAUSAL_CONV1D_FORCE_CXX11_ABI CAUSAL_CONV1D_SKIP_CUDA_BUILD
GDN_BASE_PYTHONPATH="${PYTHONPATH-}"

cat > "$GDN_ROOT/runtime_identity.py" <<'PY'
import importlib.metadata, json, platform, sys
import torch, triton
print(json.dumps({
    "python":sys.version,"machine":platform.machine(),"torch":torch.__version__,
    "torch_file":torch.__file__,"torch_cuda":torch.version.cuda,
    "torch_cxx11_abi":torch._C._GLIBCXX_USE_CXX11_ABI,
    "triton":triton.__version__,"triton_distribution":importlib.metadata.version("triton"),"triton_file":triton.__file__,
},sort_keys=True,indent=2))
assert platform.machine() in ("aarch64","arm64"), "This candidate is for Thor ARM64"
assert torch.__version__ == "2.12.0a0+5aff3928d8.nv26.05", "Base Torch differs; re-audit before building"
assert triton.__version__ == "3.7.0", "Base Triton module differs; re-audit before building"
assert importlib.metadata.version("triton") == "3.7.0+gitb4e20bb.nv26.5", "Base Triton distribution differs; re-audit before building"
assert torch.version.cuda.startswith("13.2"), "Expected NGC CUDA13.2"
PY
"$GDN_PYTHON" "$GDN_ROOT/runtime_identity.py" > "$GDN_ROOT/runtime-before.json"
"$GDN_PYTHON" -m pip list --format=freeze > "$GDN_ROOT/packages-before.freeze.txt"
nvcc --version > "$GDN_ROOT/nvcc-version.txt"
g++ --version > "$GDN_ROOT/compiler-version.txt"

# Missing helpers only, into the separate build-tools overlay. Never resolve deps.
mapfile -t GDN_NEEDED < <("$GDN_PYTHON" - <<'PY'
import importlib.util
for module, requirement in [
    ("setuptools","setuptools==80.9.0"),("wheel","wheel==0.45.1"),
    ("ninja","ninja==1.13.0"),("packaging","packaging==25.0"),("einops","einops==0.8.1"),
]:
    if importlib.util.find_spec(module) is None:
        print(requirement)
PY
)
printf '%s\n' "${GDN_NEEDED[@]}" > "$GDN_ROOT/needed-helpers.txt"
if ((${#GDN_NEEDED[@]})); then
  "$GDN_PYTHON" -m pip download --no-deps --only-binary=:all: \
    --dest "$GDN_ROOT/build-wheels" "${GDN_NEEDED[@]}"
  "$GDN_PYTHON" -m pip install --no-deps --no-index --target "$GDN_ROOT/build-tools" \
    "$GDN_ROOT"/build-wheels/*.whl
fi
export PATH="$GDN_ROOT/build-tools/bin:$PATH"
export PYTHONPATH="$GDN_ROOT/build-tools${GDN_BASE_PYTHONPATH:+:$GDN_BASE_PYTHONPATH}"

# PyPI is authoritative for archive hashes. Fixed expected hashes also reject a
# mirror/release mismatch. Download only these three tiny artifacts, not Torch.
"$GDN_PYTHON" - "$GDN_ROOT" <<'PY'
import hashlib, json, pathlib, urllib.request, sys
root=pathlib.Path(sys.argv[1]); artifacts=[]
pins=[
 ("fla-core","0.6.0","fla_core-0.6.0-py3-none-any.whl","b0708b70e0b119cd58a0a5146ea56cb175cef6ed969d8ac60267e3e077ce3ebd"),
 ("flash-linear-attention","0.6.0","flash_linear_attention-0.6.0-py3-none-any.whl","6dec9735f8454523c13553622e98aff30152dde6586b4ac7d88010fbb7a06b5f"),
 ("causal-conv1d","1.7.0","causal_conv1d-1.7.0.tar.gz","3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b"),
]
for name,version,filename,expected in pins:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json",timeout=20) as response:
        metadata=json.load(response)
    entry=next(item for item in metadata["urls"] if item["filename"]==filename)
    assert entry["digests"]["sha256"]==expected, (filename,"published hash changed")
    with urllib.request.urlopen(entry["url"],timeout=20) as response:
        blob=response.read(4*1024*1024+1)
    assert len(blob)<=4*1024*1024 and hashlib.sha256(blob).hexdigest()==expected, (filename,"payload hash mismatch")
    with (root/"wheels"/filename).open("xb") as stream:
        stream.write(blob)
    artifacts.append({"package":name,"version":version,"filename":filename,"sha256":expected,"url":entry["url"]})
(root/"source-pins.json").write_text(json.dumps({
    "fla_git_release_commit":"413ad2c655ece57b94bc9ddc51fdbd43cf09d984",
    "causal_conv1d_git_release_commit":"cd81f0413cad2fc1e6f17e785ac39f59aae690cd",
    "artifacts":artifacts},indent=2)+"\n")
PY

# Uses exactly the current NGC headers/libtorch and CXX11 ABI; bypasses guessed
# release wheels and PEP517's isolated environment (which could install Torch).
"$GDN_PYTHON" -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$GDN_ROOT/wheels" "$GDN_ROOT/wheels/causal_conv1d-1.7.0.tar.gz"
# Install both FLA namespace wheels in ONE transaction into the new empty target.
"$GDN_PYTHON" -m pip install --no-deps --no-index --target "$GDN_ROOT/site" \
  "$GDN_ROOT"/wheels/*.whl
export PYTHONPATH="$GDN_ROOT/site:$PYTHONPATH"
"$GDN_PYTHON" "$GDN_ROOT/runtime_identity.py" > "$GDN_ROOT/runtime-candidate.json"
cmp "$GDN_ROOT/runtime-before.json" "$GDN_ROOT/runtime-candidate.json"
"$GDN_PYTHON" -m pip list --format=freeze > "$GDN_ROOT/packages-candidate.freeze.txt"
"$GDN_PYTHON" - "$GDN_ROOT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
files=[*root.glob("wheels/*"),*root.glob("build-wheels/*")]
(root/"artifact-sha256.json").write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},sort_keys=True,indent=2)+"\n")
PY
cat > "$GDN_ROOT/check_imports.py" <<'PY'
import importlib.metadata as md, inspect, json, torch, triton
import causal_conv1d_cuda
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
print(json.dumps({"torch":torch.__version__,"triton":triton.__version__,
 "fla_core":md.version("fla-core"),"flash_linear_attention":md.version("flash-linear-attention"),
 "causal_conv1d":md.version("causal-conv1d"),"causal_cuda_so":causal_conv1d_cuda.__file__,
 "capability":torch.cuda.get_device_capability(),"fla_available":is_flash_linear_attention_available(),
 "conv_available":is_causal_conv1d_available(),"qwen_gdn_fast_path":q.is_fast_path_available,
 "gdn_symbols":{n:getattr(q,n).__module__ for n in ["causal_conv1d_fn","causal_conv1d_update","chunk_gated_delta_rule","fused_recurrent_gated_delta_rule"]},
 "fused_norm":inspect.getfile(FusedRMSNormGated)},indent=2))
assert torch.cuda.get_device_capability()==(11,0)
assert q.is_fast_path_available
PY
# Deliberately NOT executed: GPU import/smoke/parity must run as a separate
# candidate process with the shared Flora GPU lock, after the baseline is safe.
printf 'BUILD_COMPLETE_NOT_RUNTIME_VALIDATED\n' > "$GDN_ROOT/BUILD_STATUS.txt"
printf 'Overlay built at %s. Import check is provided but NOT executed.\n' "$GDN_ROOT"
printf 'Candidate PYTHONPATH: %s\n' "$PYTHONPATH"
