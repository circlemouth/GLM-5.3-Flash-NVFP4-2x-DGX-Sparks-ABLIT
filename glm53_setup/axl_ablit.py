"""Build and verify the pinned, offline AXL-ABLIT derived checkpoint."""

from __future__ import annotations

import ctypes
import errno
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import sys
import uuid
from pathlib import Path

from .config import ROOT

LOCK_PATH = ROOT / "config/axl_ablit.lock.json"
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "axl-ablit-manifest.json"
MAIN_LAYERS = tuple(range(15, 44))
MTP_LAYER = 45
CHUNK = 8 << 20
MAX_HEADER = 256 << 20
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
HEX64 = re.compile(r"[0-9a-f]{64}")


class AxlAblitError(ValueError):
    """The source or derived artifact violates the fixed contract."""


def load_lock() -> dict:
    lock = _json(LOCK_PATH)
    if lock.get("schema") != 1 or lock.get("output_manifest") != MANIFEST_NAME:
        raise AxlAblitError("invalid AXL-ABLIT lock schema")
    return lock


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_source_tree(root: Path, expected: dict, label: str) -> dict:
    """Authenticate every file in a pinned source tree."""
    tree = expected.get("tree")
    if (
        not isinstance(tree, dict)
        or type(tree.get("file_count")) is not int
        or type(tree.get("total_bytes")) is not int
        or not HEX64.fullmatch(str(tree.get("aggregate_sha256", "")))
    ):
        raise AxlAblitError(f"invalid {label} source-tree lock")
    rows = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if not path.is_file():
            raise AxlAblitError(f"{label} source tree contains a non-file")
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        rows.append((relative, size, sha256_file(path)))
        total += size
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    actual = {
        "file_count": len(rows),
        "total_bytes": total,
        "aggregate_sha256": aggregate.hexdigest(),
    }
    if actual != tree:
        raise AxlAblitError(f"{label} source tree does not match the pinned revision")
    return {"repo": expected["repo"], "revision": expected["revision"], "tree": actual}


def main_stem(layer: int) -> str:
    return f"model.language_model.layers.{layer}.self_attn.o_proj"


def main_keys() -> tuple[str, ...]:
    return tuple(
        f"{main_stem(layer)}.{suffix}"
        for layer in MAIN_LAYERS
        for suffix in ("weight", "weight_scale", "weight_scale_2")
    )


def mtp_key() -> str:
    return f"model.language_model.layers.{MTP_LAYER}.self_attn.o_proj.weight"


def transformed_keys() -> tuple[str, ...]:
    return (*main_keys(), mtp_key())


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise AxlAblitError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AxlAblitError(f"cannot read JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise AxlAblitError(f"JSON root must be an object: {path.name}")
    return value


def validate_axl_identity(root: Path) -> dict:
    """Require the complete base to carry the pinned public AXL identities."""
    lock = load_lock()["axl"]
    actual = {}
    for name, expected in lock["identity_sha256"].items():
        path = root / name
        if not path.is_file():
            raise AxlAblitError(f"AXL identity file is missing: {name}")
        digest = sha256_file(path)
        actual[name] = digest
        if digest != expected:
            raise AxlAblitError(f"AXL identity mismatch: {name}")
    index = _json(root / INDEX_NAME)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AxlAblitError("AXL index has no weight_map")
    if set(transformed_keys()) - set(weight_map):
        raise AxlAblitError("AXL index is missing transformed tensors")
    for key, name in weight_map.items():
        if (
            not isinstance(key, str)
            or not isinstance(name, str)
            or Path(name).name != name
            or "\\" in name
            or not name.endswith(".safetensors")
        ):
            raise AxlAblitError("AXL index contains an unsafe entry")
        if not (root / name).is_file():
            raise AxlAblitError(f"AXL shard is missing: {name}")
    identity = validate_source_tree(root, lock, "AXL")
    identity["files"] = actual
    return identity


def read_header(path: Path) -> tuple[int, dict]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw_len = stream.read(8)
            if len(raw_len) != 8:
                raise AxlAblitError(f"truncated Safetensors prefix: {path.name}")
            header_len = struct.unpack("<Q", raw_len)[0]
            if not 0 < header_len <= MAX_HEADER or 8 + header_len >= size:
                raise AxlAblitError(f"invalid Safetensors header size: {path.name}")
            raw = stream.read(header_len)
        header = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AxlAblitError(f"cannot read Safetensors header: {path.name}") from exc
    if not isinstance(header, dict):
        raise AxlAblitError(f"Safetensors header is not an object: {path.name}")
    end = 0
    for key, item in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(key, str) or not isinstance(item, dict):
            raise AxlAblitError(f"invalid Safetensors entry: {path.name}")
        offsets = item.get("data_offsets")
        shape = item.get("shape")
        dtype = item.get("dtype")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or offsets[0] != end
            or offsets[1] < offsets[0]
        ):
            raise AxlAblitError(f"invalid Safetensors tensor metadata: {key}")
        end = offsets[1]
    if 8 + header_len + end != size:
        raise AxlAblitError(f"Safetensors data length mismatch: {path.name}")
    return header_len, header


def tensor_meta(
    root: Path,
    index: dict,
    key: str,
    *,
    headers: dict[Path, tuple[int, dict]] | None = None,
) -> tuple[Path, int, dict]:
    name = index["weight_map"].get(key)
    if not isinstance(name, str):
        raise AxlAblitError(f"index is missing tensor: {key}")
    path = root / name
    if headers is None:
        header_len, header = read_header(path)
    else:
        if path not in headers:
            headers[path] = read_header(path)
        header_len, header = headers[path]
    item = header.get(key)
    if not isinstance(item, dict):
        raise AxlAblitError(f"indexed shard is missing tensor: {key}")
    return path, header_len, item


def hash_tensor(
    root: Path,
    index: dict,
    key: str,
    *,
    headers: dict[Path, tuple[int, dict]] | None = None,
) -> dict:
    path, header_len, item = tensor_meta(root, index, key, headers=headers)
    start, end = item["data_offsets"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(8 + header_len + start)
        remaining = end - start
        while remaining:
            block = stream.read(min(CHUNK, remaining))
            if not block:
                raise AxlAblitError(f"truncated tensor: {key}")
            digest.update(block)
            remaining -= len(block)
    return {
        "dtype": item["dtype"],
        "shape": item["shape"],
        "nbytes": end - start,
        "sha256": digest.hexdigest(),
    }


def _raw_tensor(root: Path, index: dict, key: str) -> tuple[bytearray, dict]:
    path, header_len, item = tensor_meta(root, index, key)
    start, end = item["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(8 + header_len + start)
        data = bytearray(stream.read(end - start))
    if len(data) != end - start:
        raise AxlAblitError(f"truncated tensor: {key}")
    return data, item


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise AxlAblitError("building requires PyTorch") from exc
    expected = load_lock()["requantizer"]["torch"]
    if torch.__version__ != expected:
        raise AxlAblitError(
            f"PyTorch version mismatch: expected {expected}, got {torch.__version__}"
        )
    return torch


def e2m1_codes(value, torch):
    """Reference E2M1 code selection, including midpoint round-to-even."""
    mid = torch.tensor([(a + b) / 2 for a, b in zip(E2M1_VALUES, E2M1_VALUES[1:])])
    absolute = value.abs()
    index = torch.bucketize(absolute, mid, right=True)
    midpoint = mid[(index - 1).clamp(min=0)]
    tie = (index > 0) & (absolute == midpoint) & (index % 2 == 1)
    index = index - tie.long()
    sign = (value < 0).to(torch.uint8) << 3
    return sign | index.to(torch.uint8)


def quant_nvfp4(weight, torch):
    """Pinned reference BF16/float [n,k] to W4A16_NVFP4 tensors."""
    if weight.dim() != 2 or weight.shape[1] % 16:
        raise AxlAblitError("NVFP4 input must be rank two with k divisible by 16")
    n, k = weight.shape
    weight = weight.float()
    global_scale = weight.abs().max().clamp_min(1e-12) / (6.0 * 448.0)
    groups = weight.view(n, k // 16, 16)
    block_scale = (
        (groups.abs().amax(-1) / (6.0 * global_scale))
        .clamp(min=2**-9, max=448.0)
        .to(torch.float8_e4m3fn)
    )
    normalized = (groups / (block_scale.float().unsqueeze(-1) * global_scale)).clamp(
        -6.0, 6.0
    )
    codes = e2m1_codes(normalized.reshape(n, k), torch)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, block_scale.contiguous(), global_scale.float().reshape(()).clone()


def _tensor_bytes(tensor, torch) -> bytes:
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _quantize_local(root: Path, index: dict, key: str, torch):
    raw, item = _raw_tensor(root, index, key)
    if item["dtype"] != "BF16" or len(item["shape"]) != 2:
        raise AxlAblitError(f"requant source is not BF16 matrix: {key}")
    expected = item["shape"][0] * item["shape"][1] * 2
    if len(raw) != expected:
        raise AxlAblitError(f"requant source byte size mismatch: {key}")
    weight = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(item["shape"])
    tensors = quant_nvfp4(weight, torch)
    del weight, raw
    return tensors


def reproduce_axl(nvidia: Path, axl: Path, output: Path) -> dict:
    """Run the mandatory all-29 byte-for-byte public AXL reproduction gate."""
    output_resolved = output.resolve(strict=False)
    if (
        output.is_symlink()
        or output.is_dir()
        or output_resolved.is_relative_to(axl.resolve())
        or output_resolved.is_relative_to(nvidia.resolve())
    ):
        raise AxlAblitError("reproduction output must be a regular file outside inputs")
    identity = validate_axl_identity(axl)
    lock = load_lock()
    nvidia_identity = validate_source_tree(nvidia, lock["nvidia"], "NVIDIA")
    torch = _torch()
    nvidia_index = _json(nvidia / INDEX_NAME)
    axl_index = _json(axl / INDEX_NAME)
    rows = []
    passed = True
    for layer in MAIN_LAYERS:
        source_key = f"{main_stem(layer)}.weight"
        tensors = _quantize_local(nvidia, nvidia_index, source_key, torch)
        for suffix, tensor, dtype in zip(
            ("weight", "weight_scale", "weight_scale_2"),
            tensors,
            ("U8", "F8_E4M3", "F32"),
        ):
            key = f"{main_stem(layer)}.{suffix}"
            expected = hash_tensor(axl, axl_index, key)
            raw = _tensor_bytes(tensor, torch)
            actual = {
                "dtype": dtype,
                "shape": list(tensor.shape),
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            match = actual == expected
            rows.append(
                {"key": key, "expected": expected, "actual": actual, "match": match}
            )
            passed = passed and match
        del tensors
        gc.collect()
    report = {
        "schema": 1,
        "kind": "axl-reproduction",
        "passed": passed,
        "tensor_count": len(rows),
        "main_layers": list(MAIN_LAYERS),
        "axl": identity,
        "nvidia": nvidia_identity,
        "requantizer": lock["requantizer"],
        "converter": {"module_sha256": sha256_file(Path(__file__))},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "tensors": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    if not passed:
        raise AxlAblitError("public AXL reproduction gate failed")
    return report


def validate_reproduction(path: Path) -> dict:
    report = _json(path)
    lock = load_lock()
    if (
        report.get("schema") != 1
        or report.get("kind") != "axl-reproduction"
        or report.get("passed") is not True
        or report.get("tensor_count") != 87
        or report.get("main_layers") != list(MAIN_LAYERS)
        or report.get("nvidia") != lock["nvidia"]
        or report.get("requantizer") != lock["requantizer"]
        or report.get("converter", {}).get("module_sha256")
        != sha256_file(Path(__file__))
        or report.get("axl", {}).get("repo") != lock["axl"]["repo"]
        or report.get("axl", {}).get("revision") != lock["axl"]["revision"]
        or report.get("axl", {}).get("tree") != lock["axl"]["tree"]
        or report.get("axl", {}).get("files") != lock["axl"]["identity_sha256"]
        or report.get("environment", {}).get("torch") != lock["requantizer"]["torch"]
    ):
        raise AxlAblitError("invalid AXL reproduction report")
    rows = report.get("tensors")
    if not isinstance(rows, list) or len(rows) != len(main_keys()):
        raise AxlAblitError("AXL reproduction report has wrong tensor keys")
    for row, key in zip(rows, main_keys()):
        if not isinstance(row, dict):
            raise AxlAblitError(f"invalid AXL reproduction tensor: {key}")
        expected = row.get("expected")
        actual = row.get("actual")
        dtype = {
            "weight": "U8",
            "weight_scale": "F8_E4M3",
            "weight_scale_2": "F32",
        }[key.rsplit(".", 1)[1]]
        if (
            row.get("key") != key
            or row.get("match") is not True
            or actual != expected
            or not isinstance(expected, dict)
            or set(expected) != {"dtype", "shape", "nbytes", "sha256"}
            or expected.get("dtype") != dtype
            or not isinstance(expected.get("shape"), list)
            or any(type(dim) is not int or dim < 0 for dim in expected.get("shape", []))
            or expected.get("nbytes")
            != math.prod(expected.get("shape", []))
            * {"U8": 1, "F8_E4M3": 1, "F32": 4}[dtype]
            or not HEX64.fullmatch(str(expected.get("sha256", "")))
        ):
            raise AxlAblitError(f"invalid AXL reproduction tensor: {key}")
    return report


def _copy_file(source: Path, target: Path) -> str:
    """Reflink a regular file where supported, otherwise make a private copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as src, target.open("xb") as dst:
            import fcntl

            fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # FICLONE
        shutil.copystat(source, target, follow_symlinks=True)
        return "reflink"
    except OSError:
        target.unlink(missing_ok=True)
        shutil.copy2(source, target, follow_symlinks=True)
        return "copy"


def _copy_tree(source: Path, target: Path) -> dict:
    methods = {"reflink": 0, "copy": 0}
    for parent, dirs, files in os.walk(source):
        dirs.sort()
        files.sort()
        relative = Path(parent).relative_to(source)
        (target / relative).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = Path(parent) / name
            if not src.is_file():
                raise AxlAblitError(
                    f"AXL source contains a non-file: {relative / name}"
                )
            if name == MANIFEST_NAME:
                raise AxlAblitError("AXL source already contains an AXL-ABLIT manifest")
            method = _copy_file(src, target / relative / name)
            methods[method] += 1
    return methods


def _donor_index(donor: Path) -> tuple[dict, dict]:
    from .runtime.weight_overlay import verify_assets

    lock = load_lock()
    report = verify_assets(
        donor,
        lock["donor"]["overlay_manifest_sha256"],
        lock["nvidia"]["revision"],
        lock["donor"]["revision"],
    )
    manifest = _json(donor / "manifest.json")
    by_layer = {int(row["layer"]): row for row in manifest["tensors"]}
    if set(by_layer) != {*MAIN_LAYERS, MTP_LAYER}:
        raise AxlAblitError("donor overlay has the wrong layer set")
    return by_layer, report


def _donor_raw(donor: Path, row: dict) -> tuple[bytearray, dict]:
    path = donor / row["file"]
    header_len, header = read_header(path)
    key = row["key"]
    item = header.get(key)
    if not isinstance(item, dict) or item.get("dtype") != "BF16":
        raise AxlAblitError(f"donor tensor is not BF16: {key}")
    start, end = item["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(8 + header_len + start)
        raw = bytearray(stream.read(end - start))
    if (
        len(raw) != end - start
        or hashlib.sha256(raw).hexdigest() != row["tensor_sha256"]
    ):
        raise AxlAblitError(f"donor tensor bytes mismatch: {key}")
    return raw, item


def _replacement_file(path: Path, raw: bytes, dtype: str, shape: list[int]) -> dict:
    path.write_bytes(raw)
    return {
        "path": path,
        "dtype": dtype,
        "shape": shape,
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _prepare_replacements(donor: Path, temp: Path) -> tuple[dict, dict, dict]:
    torch = _torch()
    by_layer, donor_report = _donor_index(donor)
    replacements = {}
    transformed = {}
    for layer in (*MAIN_LAYERS, MTP_LAYER):
        row = by_layer[layer]
        raw, item = _donor_raw(donor, row)
        expected_shape = item["shape"]
        if layer == MTP_LAYER:
            key = mtp_key()
            info = _replacement_file(
                temp / "l45-weight.bin", raw, "BF16", expected_shape
            )
            replacements[key] = info
            transformed[key] = {
                "donor_sha256": row["tensor_sha256"],
                "output_sha256": info["sha256"],
                "dtype": "BF16",
                "shape": expected_shape,
                "nbytes": info["nbytes"],
            }
            continue
        weight = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(expected_shape)
        tensors = quant_nvfp4(weight, torch)
        del weight, raw
        for suffix, tensor, dtype in zip(
            ("weight", "weight_scale", "weight_scale_2"),
            tensors,
            ("U8", "F8_E4M3", "F32"),
        ):
            key = f"{main_stem(layer)}.{suffix}"
            data = _tensor_bytes(tensor, torch)
            info = _replacement_file(
                temp / f"l{layer}-{suffix}.bin", data, dtype, list(tensor.shape)
            )
            replacements[key] = info
            transformed[key] = {
                "donor_weight_sha256": row["tensor_sha256"],
                "output_sha256": info["sha256"],
                "dtype": dtype,
                "shape": list(tensor.shape),
                "nbytes": info["nbytes"],
            }
        del tensors
        gc.collect()
    if tuple(replacements) != transformed_keys():
        raise AxlAblitError("replacement generation produced the wrong key order")
    return replacements, transformed, donor_report


def _copy_exact(stream, output, count: int, digest=None) -> None:
    remaining = count
    while remaining:
        block = stream.read(min(CHUNK, remaining))
        if not block:
            raise AxlAblitError("source tensor became truncated")
        output.write(block)
        if digest is not None:
            digest.update(block)
        remaining -= len(block)


def _rewrite_shard(
    source: Path, target: Path, replacements: dict, invariant_rows: list[tuple]
) -> None:
    header_len, header = read_header(source)
    rewritten = {}
    offset = 0
    for key, item in header.items():
        if key == "__metadata__":
            rewritten[key] = item
            continue
        replacement = replacements.get(key)
        dtype = replacement["dtype"] if replacement else item["dtype"]
        shape = replacement["shape"] if replacement else item["shape"]
        nbytes = (
            replacement["nbytes"]
            if replacement
            else item["data_offsets"][1] - item["data_offsets"][0]
        )
        rewritten[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    missing = set(replacements) - set(header)
    if missing:
        raise AxlAblitError(
            "replacement tensor not found in source shard: "
            + ", ".join(sorted(missing))
        )
    raw_header = json.dumps(rewritten, separators=(",", ":")).encode()
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    temporary = target.with_name(target.name + ".rewrite")
    with source.open("rb") as src, temporary.open("xb") as dst:
        dst.write(struct.pack("<Q", len(raw_header)))
        dst.write(raw_header)
        for key, item in header.items():
            if key == "__metadata__":
                continue
            replacement = replacements.get(key)
            if replacement:
                with replacement["path"].open("rb") as data:
                    _copy_exact(data, dst, replacement["nbytes"])
            else:
                start, end = item["data_offsets"]
                src.seek(8 + header_len + start)
                digest = hashlib.sha256()
                _copy_exact(src, dst, end - start, digest)
                invariant_rows.append(
                    (key, item["dtype"], item["shape"], digest.hexdigest())
                )
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, target)


def _aggregate_invariants(rows: list[tuple]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing any existing target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AxlAblitError("atomic no-replace rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    if (
        renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(target),
            rename_noreplace,
        )
        != 0
    ):
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise AxlAblitError("output already exists")
        raise OSError(error, os.strerror(error), target)


def build(
    axl: Path, nvidia: Path, donor: Path, reproduction: Path, output: Path
) -> dict:
    """Re-run the hard gate, then materialize an atomic derived checkpoint."""
    if output.exists() or output.is_symlink():
        raise AxlAblitError("output already exists")
    output_resolved = output.resolve(strict=False)
    if any(
        output_resolved.is_relative_to(path.resolve()) for path in (axl, nvidia, donor)
    ):
        raise AxlAblitError("output must not be inside an input directory")
    reproduction_resolved = reproduction.resolve(strict=False)
    if reproduction_resolved.is_relative_to(
        donor.resolve()
    ) or reproduction_resolved.is_relative_to(output_resolved):
        raise AxlAblitError("reproduction report must be outside donor and output")
    reproduction_report = reproduce_axl(nvidia, axl, reproduction)
    validate_reproduction(reproduction)
    axl_identity = reproduction_report["axl"]
    staging = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    temp = staging / ".axl-ablit-work"
    staging.mkdir(parents=True)
    (staging / ".incomplete").write_text("not launchable\n", encoding="utf-8")
    try:
        copy_methods = _copy_tree(axl, staging)
        temp.mkdir()
        replacements, transformed, donor_report = _prepare_replacements(donor, temp)
        index = _json(axl / INDEX_NAME)
        affected = sorted({index["weight_map"][key] for key in transformed_keys()})
        invariant_rows = []
        for name in affected:
            shard_replacements = {
                key: value
                for key, value in replacements.items()
                if index["weight_map"][key] == name
            }
            _rewrite_shard(
                axl / name, staging / name, shard_replacements, invariant_rows
            )
        # Shards with no transformed tensor are byte-identical private copies.
        all_shards = sorted(set(index["weight_map"].values()))
        unchanged_shards = []
        for name in all_shards:
            if name not in affected:
                source_sha = sha256_file(axl / name)
                output_sha = sha256_file(staging / name)
                if source_sha != output_sha:
                    raise AxlAblitError(f"unchanged shard drifted: {name}")
                unchanged_shards.append({"name": name, "sha256": source_sha})
        base_targets = {key: hash_tensor(axl, index, key) for key in transformed_keys()}
        for key, row in transformed.items():
            expected = base_targets[key]
            replacement = replacements[key]
            if (
                replacement["dtype"] != expected["dtype"]
                or replacement["shape"] != expected["shape"]
                or replacement["nbytes"] != expected["nbytes"]
            ):
                raise AxlAblitError(f"replacement contract differs from AXL: {key}")
            row["base_axl_sha256"] = expected["sha256"]
        output_shards = {name: sha256_file(staging / name) for name in all_shards}
        auxiliary = {}
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            relative = path.relative_to(staging).as_posix()
            if (
                relative in {".incomplete"}
                or relative.startswith(".axl-ablit-work/")
                or relative in all_shards
            ):
                continue
            source = axl / relative
            if not source.is_file():
                raise AxlAblitError(f"unexpected output file: {relative}")
            source_sha = sha256_file(source)
            output_sha = sha256_file(path)
            if source_sha != output_sha:
                raise AxlAblitError(f"auxiliary file drifted: {relative}")
            auxiliary[relative] = source_sha
        manifest = {
            "schema": 1,
            "kind": "axl-ablit-derived-checkpoint",
            "complete": True,
            "base": axl_identity,
            "nvidia_reproduction": {
                "source": reproduction_report["nvidia"],
                "converter_module_sha256": reproduction_report["converter"][
                    "module_sha256"
                ],
                "report_sha256": sha256_file(reproduction),
                "tensor_count": 87,
                "passed": True,
            },
            "donor": {
                **load_lock()["donor"],
                "asset_report": donor_report,
            },
            "requantizer": load_lock()["requantizer"],
            "converter": {
                "module_sha256": sha256_file(Path(__file__)),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "command_schema": "build-v1",
            },
            "transformed_key_count": len(transformed),
            "transformed_keys": transformed,
            "invariance": {
                "affected_shard_unchanged_tensor_count": len(invariant_rows),
                "affected_shard_unchanged_tensor_aggregate_sha256": _aggregate_invariants(
                    invariant_rows
                ),
                "unchanged_shards": unchanged_shards,
                "auxiliary_files": auxiliary,
            },
            "output": {
                "index_sha256": sha256_file(staging / INDEX_NAME),
                "shards": output_shards,
                "shard_sizes": {
                    name: (staging / name).stat().st_size for name in all_shards
                },
            },
            "copy_methods": copy_methods,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.rmtree(temp)
        verify(staging, base=axl, allow_incomplete=True)
        (staging / ".incomplete").unlink()
        _rename_noreplace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_manifest(
    root: Path, expected_manifest_sha256: str, *, allow_incomplete: bool = False
) -> dict:
    """Launch-time validation of the manifest and every checkpoint shard."""
    if not HEX64.fullmatch(str(expected_manifest_sha256)):
        raise AxlAblitError("expected manifest SHA-256 is invalid")
    if (root / ".incomplete").exists() and not allow_incomplete:
        raise AxlAblitError("derived checkpoint is incomplete")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise AxlAblitError("derived checkpoint manifest is missing or unsafe")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha256:
        raise AxlAblitError("derived checkpoint manifest SHA-256 mismatch")
    manifest = _json(manifest_path)
    lock = load_lock()
    if (
        manifest.get("schema") != 1
        or manifest.get("kind") != "axl-ablit-derived-checkpoint"
        or manifest.get("complete") is not True
        or manifest.get("base", {}).get("repo") != lock["axl"]["repo"]
        or manifest.get("base", {}).get("revision") != lock["axl"]["revision"]
        or manifest.get("base", {}).get("tree") != lock["axl"]["tree"]
        or manifest.get("base", {}).get("files") != lock["axl"]["identity_sha256"]
        or manifest.get("donor", {}).get("repo") != lock["donor"]["repo"]
        or manifest.get("donor", {}).get("revision") != lock["donor"]["revision"]
        or manifest.get("donor", {}).get("overlay_manifest_sha256")
        != lock["donor"]["overlay_manifest_sha256"]
        or manifest.get("donor", {}).get("asset_report", {}).get("manifest_sha256")
        != lock["donor"]["overlay_manifest_sha256"]
        or manifest.get("donor", {}).get("asset_report", {}).get("tensor_count") != 30
        or type(manifest.get("donor", {}).get("asset_report", {}).get("tensor_bytes"))
        is not int
        or manifest.get("donor", {}).get("asset_report", {}).get("tensor_bytes") <= 0
        or manifest.get("requantizer") != lock["requantizer"]
        or manifest.get("converter", {}).get("module_sha256")
        != sha256_file(Path(__file__))
        or manifest.get("converter", {}).get("command_schema") != "build-v1"
        or manifest.get("nvidia_reproduction", {}).get("source") != lock["nvidia"]
        or manifest.get("nvidia_reproduction", {}).get("converter_module_sha256")
        != manifest.get("converter", {}).get("module_sha256")
        or manifest.get("nvidia_reproduction", {}).get("passed") is not True
        or manifest.get("nvidia_reproduction", {}).get("tensor_count") != 87
        or not HEX64.fullmatch(
            str(manifest.get("nvidia_reproduction", {}).get("report_sha256", ""))
        )
        or manifest.get("transformed_key_count") != 88
        or list(manifest.get("transformed_keys", {})) != sorted(transformed_keys())
    ):
        raise AxlAblitError("invalid derived checkpoint manifest contract")
    transformed = manifest["transformed_keys"]
    dtype_sizes = {"U8": 1, "F8_E4M3": 1, "F32": 4, "BF16": 2}
    for key in transformed_keys():
        row = transformed[key]
        if not isinstance(row, dict):
            raise AxlAblitError(f"invalid transformed tensor contract: {key}")
        expected_dtype = (
            "BF16"
            if key == mtp_key()
            else {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32"}[
                key.rsplit(".", 1)[1]
            ]
        )
        shape = row.get("shape") if isinstance(row, dict) else None
        nbytes = row.get("nbytes") if isinstance(row, dict) else None
        donor_hash_key = "donor_sha256" if key == mtp_key() else "donor_weight_sha256"
        if (
            row.get("dtype") != expected_dtype
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or type(nbytes) is not int
            or nbytes != math.prod(shape) * dtype_sizes[expected_dtype]
            or not HEX64.fullmatch(str(row.get("output_sha256", "")))
            or not HEX64.fullmatch(str(row.get("base_axl_sha256", "")))
            or not HEX64.fullmatch(str(row.get(donor_hash_key, "")))
        ):
            raise AxlAblitError(f"invalid transformed tensor contract: {key}")
    index_path = root / INDEX_NAME
    if not index_path.is_file() or index_path.is_symlink():
        raise AxlAblitError("derived checkpoint index is missing or unsafe")
    index = _json(index_path)
    if sha256_file(index_path) != manifest.get("output", {}).get("index_sha256"):
        raise AxlAblitError("derived checkpoint index drifted")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AxlAblitError("derived checkpoint index has no weight_map")
    if any(
        not isinstance(key, str)
        or not isinstance(name, str)
        or Path(name).name != name
        or "\\" in name
        or not name.endswith(".safetensors")
        for key, name in weight_map.items()
    ):
        raise AxlAblitError("derived checkpoint index contains an unsafe entry")
    if set(transformed_keys()) - set(weight_map):
        raise AxlAblitError("derived checkpoint index is missing transformed tensors")
    names = sorted(set(weight_map.values()))
    output = manifest.get("output", {})
    shards = output.get("shards") if isinstance(output, dict) else None
    shard_sizes = output.get("shard_sizes") if isinstance(output, dict) else None
    if (
        not isinstance(shards, dict)
        or not isinstance(shard_sizes, dict)
        or set(names) != set(shards)
        or set(names) != set(shard_sizes)
        or any(not HEX64.fullmatch(str(value)) for value in shards.values())
        or any(type(value) is not int or value <= 0 for value in shard_sizes.values())
    ):
        raise AxlAblitError("derived checkpoint shard contract drifted")
    for name in names:
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != shard_sizes[name]
        ):
            raise AxlAblitError(f"derived checkpoint shard size drifted: {name}")
        if sha256_file(path) != shards[name]:
            raise AxlAblitError(f"derived checkpoint shard content drifted: {name}")
    evidence = manifest.get("invariance")
    auxiliary = evidence.get("auxiliary_files") if isinstance(evidence, dict) else None
    unchanged = evidence.get("unchanged_shards") if isinstance(evidence, dict) else None
    affected_count = (
        evidence.get("affected_shard_unchanged_tensor_count")
        if isinstance(evidence, dict)
        else None
    )
    if (
        not isinstance(auxiliary, dict)
        or not auxiliary
        or not isinstance(unchanged, list)
        or type(affected_count) is not int
        or affected_count < 0
        or not HEX64.fullmatch(
            str(evidence.get("affected_shard_unchanged_tensor_aggregate_sha256", ""))
        )
    ):
        raise AxlAblitError("derived checkpoint invariance contract drifted")
    for relative, digest in auxiliary.items():
        if not isinstance(relative, str):
            raise AxlAblitError("derived checkpoint auxiliary contract is unsafe")
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or candidate.as_posix() != relative
            or ".." in candidate.parts
            or "\\" in relative
            or not HEX64.fullmatch(str(digest))
        ):
            raise AxlAblitError("derived checkpoint auxiliary contract is unsafe")
        path = root / candidate
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise AxlAblitError(
                f"derived checkpoint auxiliary file drifted: {relative}"
            )
    affected_shards = {weight_map[key] for key in transformed_keys()}
    expected_unchanged = [
        {"name": name, "sha256": shards[name]}
        for name in names
        if name not in affected_shards
    ]
    if unchanged != expected_unchanged:
        raise AxlAblitError("derived checkpoint unchanged-shard evidence drifted")
    expected_files = set(names) | set(auxiliary) | {MANIFEST_NAME}
    if allow_incomplete:
        expected_files.add(".incomplete")
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AxlAblitError("derived checkpoint contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise AxlAblitError("derived checkpoint file set drifted")
    return {
        "manifest": manifest,
        "manifest_sha256": actual_manifest_sha,
        "shard_count": len(names),
    }


def verify(
    root: Path,
    *,
    base: Path | None = None,
    expected_manifest_sha256: str | None = None,
    allow_incomplete: bool = False,
) -> dict:
    """Fail closed on identity, transformed tensors, index, or invariance drift."""
    manifest_path = root / MANIFEST_NAME
    if expected_manifest_sha256 is None:
        if not manifest_path.is_file():
            raise AxlAblitError("derived checkpoint manifest is missing")
        expected_manifest_sha256 = sha256_file(manifest_path)
    launch = verify_manifest(
        root, expected_manifest_sha256, allow_incomplete=allow_incomplete
    )
    manifest = launch["manifest"]
    actual_manifest_sha = launch["manifest_sha256"]
    index = _json(root / INDEX_NAME)
    names = sorted(set(index.get("weight_map", {}).values()))
    output_headers: dict[Path, tuple[int, dict]] = {}
    transformed_actual = {}
    for key in transformed_keys():
        actual = hash_tensor(root, index, key, headers=output_headers)
        expected = manifest["transformed_keys"][key]
        if (
            actual["sha256"] != expected["output_sha256"]
            or actual["dtype"] != expected["dtype"]
            or actual["shape"] != expected["shape"]
        ):
            raise AxlAblitError(f"transformed tensor drifted: {key}")
        transformed_actual[key] = actual
    mtp_prefix = f"model.language_model.layers.{MTP_LAYER}.self_attn.o_proj."
    mtp_quant = {f"{mtp_prefix}weight_scale", f"{mtp_prefix}weight_scale_2"}
    if mtp_quant & set(index["weight_map"]):
        raise AxlAblitError("native MTP o_proj was newly quantized")
    if base is not None:
        validate_axl_identity(base)
        base_index = _json(base / INDEX_NAME)
        if index["weight_map"] != base_index["weight_map"]:
            raise AxlAblitError("derived checkpoint index mapping changed")
        base_headers: dict[Path, tuple[int, dict]] = {}
        for name in names:
            base_headers[base / name] = read_header(base / name)
            if root / name not in output_headers:
                output_headers[root / name] = read_header(root / name)
            if set(base_headers[base / name][1]) != set(output_headers[root / name][1]):
                raise AxlAblitError(f"derived checkpoint shard keys changed: {name}")
        invariant_rows = []
        transformed = set(transformed_keys())
        affected_shards = {index["weight_map"][key] for key in transformed}
        for key in transformed_keys():
            source = hash_tensor(base, base_index, key, headers=base_headers)
            expected = manifest["transformed_keys"][key]
            if source["sha256"] != expected["base_axl_sha256"]:
                raise AxlAblitError(f"base transformed tensor drifted: {key}")
            # Recomputed scales and the native MTP donor tensor can legitimately
            # be byte-identical. Every main-layer donor weight must still differ.
            if (
                key != mtp_key()
                and key.endswith(".weight")
                and source == transformed_actual[key]
            ):
                raise AxlAblitError(f"transformed weight was not changed: {key}")
        for key in sorted(index["weight_map"]):
            if key in transformed:
                continue
            source = hash_tensor(base, base_index, key, headers=base_headers)
            output = hash_tensor(root, index, key, headers=output_headers)
            if source != output:
                raise AxlAblitError(f"non-target tensor drifted: {key}")
            if index["weight_map"][key] in affected_shards:
                invariant_rows.append(
                    (key, source["dtype"], source["shape"], source["sha256"])
                )
        evidence = manifest["invariance"]
        if (
            len(invariant_rows) != evidence["affected_shard_unchanged_tensor_count"]
            or _aggregate_invariants(invariant_rows)
            != evidence["affected_shard_unchanged_tensor_aggregate_sha256"]
        ):
            raise AxlAblitError("invariance aggregate mismatch")
        for relative, digest in evidence["auxiliary_files"].items():
            if (
                sha256_file(base / relative) != digest
                or sha256_file(root / relative) != digest
            ):
                raise AxlAblitError(f"auxiliary file drifted: {relative}")
    return {
        "passed": True,
        "manifest_sha256": actual_manifest_sha,
        "transformed_key_count": 88,
        "shard_count": len(names),
        "base_invariance_checked": base is not None,
    }
