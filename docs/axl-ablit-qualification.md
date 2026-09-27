# AXL-ABLIT Derived Checkpoint Qualification Report

Evaluation date: 2026-09-27

Checkpoint generation implementation: `2d6472588742f75238e412f2a633d32447790179`

Controller evaluation integration: `feaed79e`

Scope: A pre-generated AXL-ABLIT derived checkpoint based on standard AXL

## Decision

Generation of the AXL-ABLIT derived checkpoint, integrity of all shards, the fail-closed pre-launch gate, and **real TP=2/MTP inference and speed measurements on two PGX systems passed**. Derived model B completed one warmup and three measured runs for each short, medium, and long synthetic input. Each measured run returned 64 output tokens and usage. NVIDIA controls A and A′ were measured under the same conditions before and after B; after each configuration's run series, the Controller stopped and released the inference processes on both nodes.

No separate C comparison was run with standard AXL, so the speed difference versus standard AXL remains undetermined. B was faster than the mean of the NVIDIA controls on short synthetic input, but uniform speedups were not observed for medium and long inputs. These measurements do not establish MTP acceptance rate, answer quality, or qualification for long-duration operation.

Checkpoint weights were not uploaded; the candidate was not adopted for production, and no permanent routing was added.

## Fixed inputs

| Role | Repository | Revision |
|---|---|---|
| AXL base | `Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16` | `8da41003c93593de40bf4a48e27f8d3f41973ad7` |
| NVIDIA BF16 parity source | `nvidia/GLM-5.3-Flash-NVFP4` | `423acf37583782c51c142d145aef733d72943d93` |
| Donor | `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` | `745aac2ff0f10acf961f396df3f9418598aa7327` |

The input tree was fixed by file count, total bytes, and the aggregate SHA-256 of all files. The converter SHA-256 is `3e3cd0e37706a63e5959baf0f38d7b1268b83d6e42ce48f5b7f66c98c54669ef`; PyTorch is `2.13.0+cu130`.

## Generated artifacts

- Derived checkpoint: privately retained on PGX-A (weights are not included in the repository)
- Manifest: `axl-ablit-manifest.json`
- Manifest SHA-256: `eef92679b27ed4fac41873d7aafb4f67371de4f875b74c023db9fa50d5975b3a`
- Shard count: `18`
- Total checkpoint bytes: `194881742293`
- NVIDIA-to-published-AXL reproduction report: privately retained `all-29-reproduction.json`
- Reproduction report SHA-256: `f2662f9474c650faf679d1e259285547f2ba874e42fbeb87705ea6b15651225b`

Generation used a sibling staging directory on the same filesystem, then made the output visible exactly once with Linux `renameat2(RENAME_NOREPLACE)` after validation. An existing destination is never replaced, even if it is an empty directory. On failure, staging is removed recursively; the input tree and cache are left unchanged.

## Conversion contract

Only `self_attn.o_proj` in main layers 15–43 was requantized from donor BF16 to AXL-compatible `W4A16_NVFP4`. The targets were three tensors per layer—`weight`, `weight_scale`, and `weight_scale_2`—for 87 tensors total.

At native MTP layer 45, donor provenance was assigned only to the BF16 `self_attn.o_proj.weight`. The validated donor payload was byte-identical to the AXL base, so the output digest is the same as the base. No `weight_scale` or `weight_scale_2` was added. There are 88 converted keys in total.

Layers 0–14, layer 44, all other tensors, index mapping, tokenizer, template, configuration, and auxiliary files were retained from the AXL base. Donor provenance records the tensor payload digest, not the digest of the entire safetensors file.

## Validation results

| Check | Result |
|---|---|
| Requantized main layers 15–43 from NVIDIA BF16 and compared the 87 tensors with published AXL for dtype, shape, nbytes, and bytes | `87/87` matched |
| Converted key count | `88/88` |
| MTP layer 45 dtype | `BF16` |
| No quantization scales on MTP layer 45 | Pass |
| Size and SHA-256 of all output shards | Pass |
| Index mapping and shard key set | Pass |
| All untouched tensors, auxiliary files, and invariants in affected shards | Pass |
| No symlinks, extra files, or `.incomplete` | Pass |
| Parse of non-production settings pinned to the manifest SHA-256 and derived-checkpoint pre-launch gate | Pass |
| Host test suite | `455 passed, 48 skipped, 319 subtests passed` |
| Focused CPU suite in the pinned reference container | `15 passed` |
| Additional targeted suite for Controller evaluation integration | `557 passed, 177 subtests passed` |
| Ruff and `git diff --check` | Pass |

Before making the output visible, the build ran full `verify()` against staging with the base AXL. Afterward, the pre-launch gate rechecked all shards and the file set using the actual manifest SHA-256. Within the same inspect, the verification evidence was passed to the launch identity, avoiding a duplicate rehash of approximately 195 GB.

## Controller integration

When `runtime.derived_checkpoint` specifies the generated checkpoint and manifest SHA-256, the launcher selects the derived checkpoint directly without applying the standard AXL runtime overlay. Simultaneous use of `runtime.weight_overlay` is rejected. Any mismatch in the manifest, shards, auxiliary files, or conversion contract stops launch before startup.

A dedicated temporary evaluation profile and exclusive Controller lease were used for pre-launch checks on both PGX systems, TP=2/MTP startup, authenticated smoke testing, model identity attestation, shutdown, and memory release through the Controller. A and A′ used the NVIDIA control model; B used the AXL-ABLIT derived checkpoint. Existing routes for standard BIZ, the BF16-ABLIT runtime overlay, and standard AXL were preserved.

## A→B→A′ speed measurements

Both PGX systems used image digest `sha256:69ddbd52ce81346bba435ed4373ca804927876029e845bd4cbc6d4f2fb593587`, TP=2, 3 MTP speculative tokens, and a context length of 65,536. Authenticated streaming requests were sent to the rank-zero API on PGX-A. Synthetic input targets were 32, 2048, and 8192 tokens; each condition had one warmup and three measured runs. `temperature=0`, `seed=42`, and a 64-token output limit were used; every measured response returned 64 output tokens and usage. Values below are arithmetic means of the three runs per condition. Output token/s includes the full request duration; decode token/s is measured after the first output token.

| Target input | Actual input tokens | Run | Output token/s | Decode token/s | TTFT (s) |
|---:|---:|:---|---:|---:|---:|
| 32 | 101–105 | A | 21.345 | 24.562 | 0.423 |
| 32 | 101–105 | B | 25.077 | 29.105 | 0.381 |
| 32 | 101–105 | A′ | 21.741 | 24.803 | 0.396 |
| 2048 | 3349–3369 | A | 13.233 | 28.088 | 2.588 |
| 2048 | 3349–3369 | B | 13.146 | 27.359 | 2.554 |
| 2048 | 3349–3369 | A′ | 13.272 | 28.249 | 2.586 |
| 8192 | 13262–13414 | A | 4.937 | 25.742 | 10.501 |
| 8192 | 13262–13414 | B | 5.007 | 25.144 | 10.255 |
| 8192 | 13262–13414 | A′ | 5.098 | 25.953 | 10.096 |

B's output token/s relative to the mean of A and A′ was +16.40% for short input, −0.80% for medium input, and −0.20% for long input. Actual input token counts were approximately 3.22×, 1.64×, and 1.63× the target lengths for short, medium, and long inputs, respectively. With only three measurements per condition, no speedup beyond short inputs is inferred.

Across the before-and-after snapshots for each configuration, the increments in CUDA errors, OOMs, and service crashes were all **0**. A′'s cumulative OOM count was 4 both before and after; no new OOM was attributable to the A′ measurements. The MTP setting and startup state were confirmed, but the measurement tool did not record speculative-token acceptance rate. Speed measurements alone do not establish answer quality or long-term stability.

Individual measurements and before/after snapshots were retained privately and are not included in the repository. SHA-256 values for `A-speed.json`, `B-speed.json`, and `A-prime-speed.json` are `d354410e80d23961927531acb450074f580f3d16bdbcbc649523579f7252705a`, `caa5dc1e1e799c3809ea134982d4b48ae333c6175a791b0327e0fb0e6d8ddf13`, and `28a00b4a3b8bc04aba6a5062dc25fa49ff281a4480852385331e539229a9e81d`, respectively. The dedicated profile was `glm53-axl-ablit-cluster-64k`; the derived model fingerprint was `fc08aebea8c3cb03538a124c7a59437ac5067aeed60bf179108cbf30d923963f`. An independent recalculation of output speed by `gpt-6-luna` (reasoning effort `max`) confirmed alignment by sample index and actual input-token count.

## Independent review

An independent static review was performed by `gpt-6-luna` at reasoning effort `max`. The review identified a publication race with an existing destination and a duplicate manifest hash within the same inspect; both were fixed and marked `resolved` in follow-up review. The distinction between the real donor manifest's file digest and tensor payload digest was also corrected and marked `resolved` in an additional follow-up review. No unresolved P0/P1/P2 issues remain.

## Controller restoration

The initial logical state was `single-prod` / `single-prod-ablit` / `SINGLE_PROD`, with admission open, no leases, and no warnings. The evaluation state `emergency-evo-only` was returned to `single-prod-ablit` through the standard Controller transition. Read-back at revision `1769` confirmed the same logical state, admission open, no leases, and no warnings.

Immediately after restoration, Controller doctor reported `ok=true`. The active route and actual profile, both nodes' swap and tunnel, model health, NCCL, synthetic-input smoke test, and general LiteLLM path all passed critical checks. `deepseek_loopback` and `laguna_loopback` were `false` for nonessential checks on this profile, while the in-scope `active_profile_model_health` and `active_profile_smoke` were `true`. The revision was not rolled back to `1746` because its number advances with state transitions.

## Qualification boundary

This checkpoint is a local validation artifact; its weights were not uploaded, and it was not adopted for production. Real inference and output speed with TP=2/MTP configured were confirmed, but the evaluation does not infer speculative-token acceptance rate, answer quality, long-duration stability, or performance versus standard AXL. Disabling safety checks, new kernels, training, changing the donor, whole-model requantization, and mixed BF16 29-module serving were outside the scope of this evaluation.
