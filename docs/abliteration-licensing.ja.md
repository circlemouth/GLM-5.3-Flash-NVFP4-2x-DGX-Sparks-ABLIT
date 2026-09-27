# abliteration派生モデルの出所とライセンス

<!-- Modified for static AXL-ABLIT distribution preparation; upstream text retained. -->

このforkは、BIZのApache-2.0の`LICENSE`、`NOTICE`、`LICENSES/`、第三者通知を保持します。
新しいoverlayコードにはApache-2.0のSPDX表示を付けています。
改変対象のvLLM loaderは元のApache-2.0表示を残し、このforkによる変更を明示します。
既存のMITとApacheの表示は、各構成要素に引き続き適用されます。

重みはコードとは別に運用者が取得・生成する資産であり、このGitソース配布には含めません。
以下の別配布に向けた準備方針は、重みをアップロード済みとする告知ではありません。
固定revisionの[NVIDIAモデルカード](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/423acf37583782c51c142d145aef733d72943d93/README.md)はNVFP4 checkpointをMITと表示し、商用と非商用の利用に言及しています。
同snapshotには独立したLICENSEファイルがありません。
NVIDIAが親モデルとして示す[Z.AIのGLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)にはMITのLICENSEがあります。
重みを再配布する場合は、該当する著作権表示、許諾文、免責文を保持する必要があります。

任意のdonorは[Dealignの固定revision](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4/tree/745aac2ff0f10acf961f396df3f9418598aa7327)です。
その[LICENSE](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4/blob/745aac2ff0f10acf961f396df3f9418598aa7327/LICENSE)はMITで、Z.AIの著作権表示を含みます。
モデルカードもMITと表示しています。
一方、編集された各テンソルの由来を最後まで追える資料はモデルカードにありません。
このforkは、すべての上流の権利と再配布時の表示条件が解決済みとは認定しません。
ローカルの抽出manifestにdonorのcommitとテンソルのハッシュを記録します。

方式の参考となったMiaAI-Labの公開レシピは[READMEの比較表](../README.ja.md#dgx-spark向けの他のglm-53-flashレシピ)に示します。
Miaのコード、script、EXL3 runtime、モデルファイル、test、文書本文は取り込んでいません。
この機能にAGPL由来のコードは追加していません。
AGPL自体を商用利用禁止とは扱いません。

overlayはソフトウェアとモデルの実験であり、法務、医療、安全、臨床の認証ではありません。
運用者はソースのライセンス、モデルの条件、利用する環境に応じた義務を確認してください。
ビルド後のコンテナ内の依存物には、それぞれの条件が適用されます。

## 静的AXL-ABLIT重みの別配布

静的チェックポイントは、任意のBF16 runtime overlayとは別の成果物です。入力の正本は[AXL-ABLIT lock](../config/axl_ablit.lock.json)であり、公開AXL、NVIDIAの再現確認用ソース、Dealignドナーを指定します。変換契約は[派生チェックポイント設計](axl-ablit-derived-checkpoint.ja.md)、生成物の識別情報と実測結果は[検証報告](axl-ablit-qualification.ja.md)を正本とします。処理対象のキーがすべてバイト変更されたとは限らず、報告された成果物のnative MTPドナーはAXLと同一のバイト列です。

固定された入力モデルはMIT条件を示しています。これらの入力との一致が検証された成果物について、上流の著作権表示、許諾文、免責文を保持したMITの重み配布を準備します。NVIDIAとAXLのモデルカード、DealignのモデルカードとLICENSEも来歴資料として保存します。上流カードの性能値を、この派生モデルの実測値として扱いません。GitのソースコードはApache-2.0のままであり、変換ツールの条件だけを理由に出力重みを別ライセンスへ変更しません。追加材料を含める場合は、その条件を別途確認します。

数値変換の参照元はlockに記録されています。そのApache-2.0原文は[LICENSES](../LICENSES/tenhkspark-Apache-2.0.txt)、帰属は[第三者通知](../THIRD_PARTY_NOTICES.md)に保持します。この参照元表示は、すべての関数を複製したという認定ではありません。MITの許諾には適用される通知の保持が必要であり、上流の権利関係すべてに対する保証ではありません。

## 公開用文書はチェックポイントの外側に置く

派生チェックポイントの検証は、記録された補助ファイルのハッシュ、再帰的なファイル一覧の完全一致、変換モジュールのdigestを確認します。チェックポイント内へのLICENSE追加、継承READMEの置換、コメント追記だけの変換器変更でも既存成果物の検証に失敗し得ます。ライセンス整備のために生成時manifestを変更したり、検証を緩めたり、モデルを再生成したりしません。

別配布では、外側のディレクトリ直下に公開モデルカード、MIT原文、上流通知、審査済みの公開来歴を置き、元のチェックポイントのバイト列と相対ファイル名を`checkpoint/`配下に保持する構成を選べます。ランタイムのderived-checkpoint pathは外側ではなくこの子ディレクトリを指します。ダウンローダーのキャッシュや追加文書は中に置きません。これは配布物の配置案であり、新規デプロイの認定やHub直下からの汎用ロードを保証するものではありません。

公開前に、ローカル成果物と生成証跡を照合し、固定revisionの通知原文を保存し、元manifestに非公開情報がないか確認したうえで、無変更のチェックポイントを既存の方法で検証します。公開用要約はランタイム用manifestの代用品ではありません。元manifestを安全に公開できない場合は、ハッシュ固定されたファイルを黙って匿名化せず、その配布方式を停止します。外部公開は別途許可を受ける操作です。
