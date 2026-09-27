# Optional BF16 o_proj overlay

This unofficial BIZ fork can replace 29 main-model `self_attn.o_proj.weight` tensors (layers 15–43) and the native MTP tensor (layer 45) while loading NVIDIA's pinned NVFP4 checkpoint. Layer 44, layers 0–14, other attention weights, experts, vision, tokenizer, template and the API contract remain unchanged. The option is disabled unless `runtime.weight_overlay.enabled = true` is set. The fork does not distribute weights or a container image.

The donor is the immutable Dealign revision in the [asset manifest](../config/abliteration.lock.json). [Provenance and terms](abliteration-licensing.md) and [validation status](abliteration-validation.md) have separate owners. The original BIZ acceptance does not qualify this optional profile.

## Prepare the assets

Use the pinned revisions in `config/runtime.lock.json` and `config/abliteration.lock.json`. The extraction command reads the actual model indexes and Safetensors headers, downloads only the 30 named BF16 byte ranges, verifies HTTP partial responses and writes one Safetensors file per tensor under a lock. Its `manifest.json` and files stay outside Git.

```sh
python tools/prepare_abliteration.py --help
python tools/prepare_abliteration.py inspect --output /private/overlay-assets
python tools/prepare_abliteration.py extract --output /private/overlay-assets
sha256sum /private/overlay-assets/manifest.json
```

Run the same extraction or a verified, resumable transfer on both model hosts. Check `manifest.json` and all 30 file hashes on each host before launch. Do not copy the donor's `config.json`, tokenizer or chat template into the NVIDIA snapshot. `tools/prepare_mtp_view.py` still creates the existing read-only metadata view when MTP is enabled.

## Select a profile

The standard `examples/server.example.toml` has no overlay table and never opens donor files. For an overlay profile, keep the original settings and add this table beneath `[runtime]`, replacing the placeholders with the verified local path and exact digest. Give the overlay a distinct `api.served_model_name` and an image ID built from this fork's reviewed commit on both hosts.

```toml
[runtime.weight_overlay]
enabled = true
path = "/private/overlay-assets"
donor_revision = "745aac2ff0f10acf961f396df3f9418598aa7327"
manifest_sha256 = "<64 lowercase hexadecimal characters>"
```

The launcher mounts the directory read-only at `/weight-overlay`. The profile fingerprint includes the overlay setting, donor revision, manifest digest and pinned NVIDIA revision; the launch identity also records the inspected image ID. Preflight verifies the assets and rejects the derived AXL checkpoint. Switch the two ranks together through the existing [recoverable switch](launch-safety.md#all-rail-checks-and-two-rank-switch). A switch restarts both ranks, so KV and prefix-cache state is not reused across profiles. Native MTP loads layer 45 only when MTP is enabled; a disabled MTP profile applies 29 tensors, not 30.

The source-pinned image patch refuses an unknown vLLM model or MTP loader hash. At load, an absent, duplicated, changed, non-BF16 or wrong-shaped target fails the launch. It never edits the source checkpoint. Startup success alone does not prove the resulting rank-local parameters; use the [validation protocol](abliteration-validation.md) before accepting a deployment.

## Static AXL-ABLIT checkpoint

The static route is separate from the BF16 runtime overlay. It starts from the complete pinned public AXL checkpoint, requantizes donor BF16 `o_proj.weight` for main layers 15–43 to byte-exact `W4A16_NVFP4`, and copies donor BF16 only for native MTP layer 45. It does not quantize MTP and it does not reuse the runtime overlay when serving. The immutable inputs and converter are pinned in [`config/axl_ablit.lock.json`](../config/axl_ablit.lock.json).

The mandatory first command compares all 87 public AXL tensors (29 packed weights, block scales and global scales) with tensors regenerated from the pinned NVIDIA BF16 source. Every dtype, shape and raw-byte digest must match. The report pins the exact converter-module digest. `build` does not trust a supplied report: it always reruns the complete gate with its NVIDIA and AXL inputs and atomically replaces the report path before materializing any output.

```sh
python tools/prepare_axl_ablit.py reproduce \
  --nvidia /models/nvidia-pinned \
  --axl /models/axl-pinned \
  --output /private/axl-reproduction.json

python tools/prepare_axl_ablit.py build \
  --axl /models/axl-pinned \
  --nvidia /models/nvidia-pinned \
  --donor-overlay /private/overlay-assets \
  --reproduction-report /private/axl-reproduction.json \
  --output /models/axl-ablit-derived

python tools/prepare_axl_ablit.py verify \
  --checkpoint /models/axl-ablit-derived \
  --base /models/axl-pinned
```

`build` never writes the base or donor. It uses a private copy or reflink, rewrites affected shards into new files, verifies the complete artifact, and only then renames an explicitly incomplete staging directory to the requested output. Its manifest records source revisions, converter/toolchain identity, the exact 88-key delta, output shard hashes, auxiliary-file equality and invariance evidence. Preserve the generated report and manifest outside Git; neither model weights nor private runtime paths belong in this repository.

Serve the result through the existing `[runtime.derived_checkpoint]` table, never together with `[runtime.weight_overlay]`. Add the generated manifest digest to the AXL profile and use a distinct served model name:

```toml
[runtime.derived_checkpoint]
enabled = true
path = "/models/axl-ablit-derived"
manifest_sha256 = "<generated manifest SHA-256>"
requant_target = "l"
overlays = [
  # Use the same pinned KDA and MLA source overlays as server.axl.example.toml.
]

[api]
served_model_name = "glm-5.3-flash-axl-ablit"
```

When `manifest_sha256` is present, preflight hashes every shard and rejects an incomplete artifact, a wrong manifest, source/donor/requantizer provenance drift, index drift, a wrong transformed-key contract, or missing, wrong-sized, or content-modified shards. Launch identity records both the actual manifest digest and its declared artifact identity. Omitting the field retains compatibility with the pre-existing normal AXL route; it does not qualify an artifact as AXL-ABLIT.
