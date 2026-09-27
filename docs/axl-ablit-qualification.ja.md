# AXL-ABLIT 派生チェックポイント適格性報告

評価日: 2026-09-27

チェックポイント生成実装: `2d6472588742f75238e412f2a633d32447790179`

Controller 評価統合: `feaed79e`

対象: AXL 通常版を基盤とする、事前生成 AXL-ABLIT 派生チェックポイント

## 判定

AXL-ABLIT 派生 checkpoint の生成、全 shard の完全性、起動前 fail-closed ゲート、および **PGX 2台の TP=2/MTP 実推論と速度測定は合格**した。派生モデル B は短・中・長の合成入力で各1回の warmup と各3回の測定を完走し、各測定で64 token の出力と usage を得た。前後に NVIDIA 対照 A/A′ を同条件で測り、各系の終了後に Controller が両ノードの推論プロセスを停止・解放した。

通常 AXL を別途起動する C 比較は省略した。よって**通常 AXL との速度差は未判定**である。短い合成入力では B が NVIDIA 対照平均より速かったが、中・長入力で一様な高速化は確認できない。MTP の受理率、回答品質、長時間運用の適格性は本測定では示さない。

派生 checkpoint の重みは配布せず、本番採用や恒久的な routing 追加もしていない。

## 固定した入力

| 役割 | repository | revision |
|---|---|---|
| AXL 基盤 | `Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16` | `8da41003c93593de40bf4a48e27f8d3f41973ad7` |
| NVIDIA BF16 parity source | `nvidia/GLM-5.3-Flash-NVFP4` | `423acf37583782c51c142d145aef733d72943d93` |
| donor | `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` | `745aac2ff0f10acf961f396df3f9418598aa7327` |

入力 tree はファイル数、総 byte 数、全ファイル集約 SHA-256 で固定した。converter は SHA-256 `3e3cd0e37706a63e5959baf0f38d7b1268b83d6e42ce48f5b7f66c98c54669ef`、PyTorch は `2.13.0+cu130` である。

## 生成物

- 派生 checkpoint: PGX-A の非公開ローカル領域（重みはリポジトリに含まない）
- manifest: `axl-ablit-manifest.json`
- manifest SHA-256: `eef92679b27ed4fac41873d7aafb4f67371de4f875b74c023db9fa50d5975b3a`
- shard 数: `18`
- checkpoint 総 byte 数: `194881742293`
- NVIDIA→公開 AXL 再現 report: 非公開の `all-29-reproduction.json`
- 再現 report SHA-256: `f2662f9474c650faf679d1e259285547f2ba874e42fbeb87705ea6b15651225b`

生成時は同一 filesystem の sibling staging を使い、検証完了後に Linux `renameat2(RENAME_NOREPLACE)` で一度だけ公開した。既存の出力先は、空 directory であっても置換しない。失敗時は staging を再帰的に除去し、入力 tree と cache は変更しない。

## 変換契約

main layer 15–43 の `self_attn.o_proj` だけを donor BF16 から AXL 互換 `W4A16_NVFP4` へ再量子化した。各 layer の対象は `weight`、`weight_scale`、`weight_scale_2` の3 tensor、合計87 tensor である。

native MTP layer 45 は `self_attn.o_proj.weight` の BF16 だけを donor provenance へ置換した。認証済み donor payload は AXL 基盤と byte-identical だったため、出力 digest は基盤と同じである。`weight_scale` と `weight_scale_2` は追加していない。変換 key は合計88個となる。

layer 0–14、layer 44、その他の tensor、index mapping、tokenizer、template、設定、補助ファイルは AXL 基盤から維持した。donor provenance には safetensors ファイル全体の digest ではなく、tensor payload の digest を記録する。

## 検証結果

| 検証 | 結果 |
|---|---|
| NVIDIA BF16 から main layer 15–43 を再量子化し、公開 AXL の87 tensor と dtype・shape・nbytes・bytes を比較 | `87/87` 一致 |
| 変換 key 数 | `88/88` |
| MTP layer 45 の dtype | `BF16` |
| MTP layer 45 の量子化 scale 非存在 | 合格 |
| 全 output shard の size・SHA-256 | 合格 |
| index mapping と shard key set | 合格 |
| 全非対象 tensor、補助ファイル、影響 shard 内 invariant | 合格 |
| symlink、余分なファイル、`.incomplete` の不存在 | 合格 |
| manifest SHA-256 を固定した非本番設定の parse と派生 checkpoint 起動前 gate | 合格 |
| host test suite | `455 passed, 48 skipped, 319 subtests passed` |
| pinned reference container の focused CPU suite | `15 passed` |
| Controller 評価統合の追加 targeted suite | `557 passed、177 subtests passed` |
| Ruff、`git diff --check` | 合格 |

build は公開前 staging に対して base AXL を伴う完全 `verify()` を実行した。公開後は、実 manifest SHA-256 を指定した起動前 gate で全 shard と file set を再確認した。同じ inspect 内では、その検証 evidence を launch identity へ渡し、約195 GBの再 hash を重複させない。

## Controller 統合

`runtime.derived_checkpoint` に生成済み checkpoint と manifest SHA-256 を指定すると、launcher は通常の AXL runtime overlay を重ねず、派生 checkpoint を直接選ぶ。`runtime.weight_overlay` との同時指定は拒否する。manifest、shard、補助ファイル、変換契約のいずれかがずれれば起動前に停止する。

専用の一時評価 profile と排他的な Controller lease を用意し、両 PGX の起動前検査、TP=2/MTP 起動、認証付き smoke、model identity attestation、停止とメモリ解放を Controller 経由で行った。A と A′ は NVIDIA 対照モデル、B は AXL-ABLIT 派生 checkpoint を使用した。通常 BIZ、BF16-ABLIT runtime overlay、通常 AXL の既存経路は維持した。

## A→B→A′ 速度実測

両 PGX で image digest `sha256:69ddbd52ce81346bba435ed4373ca804927876029e845bd4cbc6d4f2fb593587`、TP=2、MTP speculative tokens 3、context 65,536 を固定し、PGX-A の rank-zero API へ認証付き streaming 要求を送った。合成入力の目標長は 32、2048、8192 token、各条件で warmup 1 回と測定 3 回を実行した。`temperature=0`、`seed=42`、出力上限 64 token とし、測定対象の全応答で出力 64 token と usage を確認した。表中の値は各条件の測定 3 回の算術平均で、出力速度はリクエストの終了までを含む値、decode 速度は初回出力後の速度である。

| 目標入力 | 実際の入力 token | 系 | 出力 token/s | decode token/s | TTFT 秒 |
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

B の出力 token/s は A と A′ の平均に対して短入力で +16.40%、中入力で −0.80%、長入力で −0.20% だった。実際の入力 token 数は目標値と異なり、短入力では約3.22倍、中入力では約1.64倍、長入力では約1.63倍である。各条件の測定数は3回に限られるため、短入力以外への速度向上を推定しない。

各系の測定前後で CUDA error、OOM、service crash の**増分はいずれも0**だった。A′ の OOM 累積値は前後とも4であり、A′ の測定に起因する新規 OOM ではない。MTP の設定値と起動状態は確認したが、この測定器は speculative token の受理率を記録していない。回答品質、長時間安定性も速度測定だけでは判定しない。

個別の測定・前後スナップショットは非公開のローカル記録として保存し、リポジトリには含めない。`A-speed.json`、`B-speed.json`、`A-prime-speed.json` の SHA-256 はそれぞれ `d354410e80d23961927531acb450074f580f3d16bdbcbc649523579f7252705a`、`caa5dc1e1e799c3809ea134982d4b48ae333c6175a791b0327e0fb0e6d8ddf13`、`28a00b4a3b8bc04aba6a5062dc25fa49ff281a4480852385331e539229a9e81d`。専用 profile は `glm53-axl-ablit-cluster-64k`、派生モデルの fingerprint は `fc08aebea8c3cb03538a124c7a59437ac5067aeed60bf179108cbf30d923963f`。出力速度の独立再集計は `gpt-6-luna`（推論強度 `max`）が同一 sample index と実際の入力 token 数に対応付けて確認した。

## 独立レビュー

`gpt-6-luna`、reasoning effort `max` による独立静的レビューを実施した。指摘された既存出力先との publication race と、同一 inspect 内の重複 manifest hash は修正し、追跡レビューで `resolved` を確認した。実 donor manifest の file digest と tensor payload digest の区別も修正し、追加の追跡レビューで `resolved` を確認した。未解決の P0/P1/P2 はない。

## Controller 復元

開始時の論理状態は `single-prod` / `single-prod-ablit` / `SINGLE_PROD`、admission open、lease なし、warnings なしだった。評価中の `emergency-evo-only` から正規 Controller 遷移で `single-prod-ablit` へ戻し、revision `1769` で同じ論理状態、admission open、全 lease なし、warnings なしを read-back した。

復元直後の Controller doctor は `ok=true`。active route と実際の profile、両ノードの swap・tunnel、モデル健康、NCCL、合成入力 smoke、一般 LiteLLM 経路がすべて critical check で合格した。`deepseek_loopback` と `laguna_loopback` はこの profile の非必須チェックで `false` だが、稼働対象の `active_profile_model_health` と `active_profile_smoke` は `true` である。revision の数値は状態遷移により進むため、開始時の `1746` へ巻き戻していない。

## 適格性の境界

この checkpoint はローカル検証成果物であり、重みは配布せず、本番採用もしていない。TP=2/MTP を設定した実推論と出力速度は確認したが、speculative token の受理率、回答品質、長時間安定性、通常 AXL との性能差を推定で補わない。安全チェックの解除、新規 kernel、学習、donor 変更、whole-model requantization、混在 BF16 29-module serving は本評価の対象外である。
