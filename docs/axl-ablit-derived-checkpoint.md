# AXL-ABLIT derived checkpoint design

Status: implementation design; not a production-adoption record.

## Fixed inputs and gate

- Base: `Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16` revision `8da41003c93593de40bf4a48e27f8d3f41973ad7`.
- NVIDIA source used only for the AXL reproduction gate: revision `423acf37583782c51c142d145aef733d72943d93`.
- Donor: `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` revision `745aac2ff0f10acf961f396df3f9418598aa7327`.
- Requantizer: tenhkspark commit `8ee63a676c6e8550f49b52aec7c5c05b56eb5553`, source SHA-256 `0a59e2516cc408b400bc85ffc13a4a8bfedcdd4003466260a94258625feb8046`.
- Before donor conversion is accepted, quantizing NVIDIA BF16 for all main layers 15 through 43 must reproduce each public AXL `o_proj.{weight,weight_scale,weight_scale_2}` tensor exactly in dtype, shape, and raw bytes. Representative KDA and MLA checks precede the complete gate.

## Artifact construction

Build from a complete immutable AXL snapshot into a sibling staging directory. Never mutate a cache or source snapshot. Copy non-weight files and unchanged shards with reflink when available, ordinary copy otherwise. Rewrite each affected shard as a new file, replacing exactly 29 main-layer triplets with donor-derived `W4A16_NVFP4` tensors and replacing only native MTP layer 45 `o_proj.weight` with donor BF16. Atomically rename the verified staging directory to its final path, and remove the staging directory on failure.

`build` reruns the complete reproduction gate itself; it never trusts a supplied report. The AXL and NVIDIA inputs are authenticated by pinned whole-tree aggregates, including every weight shard. The reproduction report pins the exact converter-module digest. The output manifest pins every source and converter identity, records the exact transformed key set, source and output tensor hashes, shard hashes, invariant counts, command schema, toolchain identity, and completion state. Verification rejects partial artifacts, unknown deltas, missing or duplicate index keys, dtype/shape drift, a newly quantized MTP layer, and all non-target tensor or auxiliary-file changes.

## Launcher integration

Serve only the pre-generated output through `runtime.derived_checkpoint`; keep `runtime.weight_overlay` mutually exclusive. Add an optional fixed manifest SHA-256 to the derived-checkpoint schema so existing BIZ, BF16-ABLIT, and normal AXL profiles continue to validate unchanged. When configured, preflight validates the manifest, hashes every checkpoint shard and auxiliary file against it, and rejects an unexpected file or symlink; launch identity includes the manifest digest and declared artifact identity. Existing AXL source overlays remain explicit read-only mounts.

## Tests and qualification

- Dependency-light CPU tests for strict Safetensors/index/manifest validation, atomic completion, allowed-delta and invariance rules, and launcher rejection paths.
- Torch CPU golden tests for E2M1 tie handling, even-low-nibble packing, E4M3 block scales, all-zero handling, scalar F32 global scale, and repeated-byte determinism.
- Full 29-layer public AXL reproduction report, donor conversion report, and checkpoint invariance report.
- Controller-owned exclusive TP=2/MTP load and rank-local loaded-value checks, then A→B→A′ plus normal AXL C comparison using fixed synthetic Contract 5 inputs and performance metrics.
- Restore the captured initial controller profile and verify routing, service identity, health, leases, and residual jobs.

Astra pre-review was explicitly attempted but the session rejected `openai-codex/gpt-6-astra` as an unavailable route. No substitute is used.
