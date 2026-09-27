# AXL-ABLIT派生checkpointの設計

## 固定入力と変更範囲

- 基盤: `Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16` revision `8da41003c93593de40bf4a48e27f8d3f41973ad7`
- AXL再現gateだけに使うNVIDIA source: revision `423acf37583782c51c142d145aef733d72943d93`
- donor: `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` revision `745aac2ff0f10acf961f396df3f9418598aa7327`
- 再量子化reference: tenhkspark revision `8ee63a676c6e8550f49b52aec7c5c05b56eb5553`

本体第15～43層の`self_attn.o_proj` 29組だけを、donor BF16からAXLとbyte-exactな`W4A16_NVFP4`へ再量子化します。標準MTP第45層は`o_proj.weight`だけをdonor BF16で置換し、量子化しません。第0～14層、第44層、その他のtensor、tokenizer、template、設定、構造は基盤AXLから変更しません。

## 生成と証拠

`build`は既存レポートを信用せず、固定NVIDIA入力から公開AXLの29層・87テンソルを毎回再現します。AXLとNVIDIAの入力tree全体をweight shardを含む固定集約値で認証します。dtype、形状、raw byteのいずれかが一致しない場合は失敗します。

出力manifestは、入力と変換器の識別情報、正確な変換key集合、入力・出力tensor hash、shard hash、不変性の件数、command schema、toolchain、完了状態を固定します。検証は部分生成物、不明な差分、index keyの欠落・重複、dtype／形状の変化、MTPの誤量子化、対象外tensorまたは補助fileの変化を拒否します。

生成先と同階層に未完了marker付きの一時directoryを作り、完全検証の合格後にだけmarkerを外してatomic renameします。失敗時は一時directoryを削除します。入力cacheやdonorへ書き込まず、hardlinkも使いません。

## 起動

配信には事前生成した出力だけを`runtime.derived_checkpoint`で指定し、`runtime.weight_overlay`と同時に有効化しません。AXL-ABLIT profileにはmanifest SHA-256を固定します。preflightはmanifest、全checkpoint shard、補助fileをhash検証し、想定外fileとsymlinkを拒否したうえで、launch identityへmanifest digestとartifact identityを含めます。通常BIZ、BF16-ABLIT、通常AXLの既存経路は維持します。

## 検証境界

- strictなSafetensors／index／manifest検証、atomic completion、許可差分、不変性、launcher拒否経路のCPU test
- 29層・87テンソルの公開AXL完全再現gate
- donor変換reportとcheckpoint全体の不変性report
- 利用可能な場合に限る、Controller管理下でのTP=2／MTP synthetic A/B比較

公開、upload、本番採用、安全検査の解除はこの設計に含みません。独立Astra reviewは利用可能性が確認できない場合、代替せず未実施として記録します。
