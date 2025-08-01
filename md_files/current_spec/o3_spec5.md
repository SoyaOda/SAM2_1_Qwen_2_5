承知しました。train\_minimal.py実行時に2件目のサンプル処理でNaNが発生する原因について、使用しているloss関数の実装（focal lossやdice lossなど）や初期化手順、mask\_headやseg\_queryの挙動、sam2\_hiera\_large.ptの影響も含めて精査します。また、LoraもしくはQLora導入が安定化に寄与するか、現時点でのQwen2.5-VL-3Bに適したLoRA設定についてもリサーチし、将来的にQwen2.5-VL-72Bへ拡張可能な視点で整理・提案します。

完了次第、詳細な原因分析と修正案を提示いたします。


# 問題の原因分析と修正指針

## NaN発生の原因分析

**1. \*\* **損失計算の数値不安定性:** 提示されたログより、学習1サンプル目では `focal_loss`（フォーカル損失）と`dice_loss`（ダイス損失）を足した総計が正常に計算されていますが、2サンプル目でNaNが発生しています。これは、おそらく**フォーカル損失の実装上の数値不安定性**によるものです。フォーカル損失は通常Binary Cross Entropy (BCE)に基づく損失ですが、予測確率が**完全に0または1に飽和した場合\*\*、`log(0)`や`log(1-1)`の計算でマイナス無限大（-∞）が生じ、結果としてNaNになります。特に、1サンプル目終了時のモデル更新で2サンプル目の予測が極端な値（例えば背景を**完全に0**に予測）になった場合、正解マスク上の正例ピクセル（値1）の箇所で`-log(pred)`が`-log(0)`となりフォーカル損失が∞となってNaNに繋がります。この挙動は**クラス不均衡**が大きいセグメンテーションで起きやすく（背景が大部分を占めるのでモデルが背景=0出力に固執しがち）、**フォーカル損失の安定化処理**がないと発生します。

\*\*2. \*\* **Dice損失のゼロ割り:** ダイス損失もまた注意が必要です。ダイス係数は分母に予測マスクと真のマスクの和を含みます。通常、\*\*平滑化項（ε）\*\*を加えて0除算を防ぐ実装が推奨されます。現在の実装でεが十分考慮されていない場合、**予測と真値がともに全ゼロ**といった特殊ケースで`0/0`が発生しNaNになり得ます（FoodLMMの実装でもダイス損失にはBCEと合わせて安定化が図られています）。今回のケースでは2サンプル目の真値マスクに少量の1が含まれているため、分母=0の事態ではなさそうですが、**予測マスクが完全0**になるとダイス損失は `(2*0)/(|P|+|T|)` = 0 となるだけでNaNにはならないはずです。従って主原因はフォーカル損失側と推測できます。

**3. \*\* **mask\_headの初期化方法:** ログを見ると、**mask\_head（マスク出力層）が各サンプル処理前に再初期化**されています。初期出力のマスク統計が`min=0.5, max=0.5`と全画素一様0.5になっていたのは、おそらく**mask\_headの重みが初期状態でバイアス=0、重み=0**近傍であるためです。すなわち**入力に依存せず0ロジットを出力し、シグモイド後0.5となる**状態でした。この挙動だと、**mask\_headの重みが学習されない限り**どの画像でも常に0.5予測になり、モデルはseg\_queryなど前段の出力をいくら変化させてもマスク出力を変えられません（mask\_headの重みが0なら勾配も伝播しません）。実際、mask\_headを毎回再初期化していては**重みが蓄積的に学習されない**ため、1サンプル目終了後の学習で変化したのはmask\_head以外（例えばSAMのデコーダ側の出力や統合トークン空間関連）になります。その結果、2サンプル目でのmask\_head出力は依然0.5中心でしょうが、学習の副作用で**seg\_queryや他の層が異常値\*\*（例えば非常に大きな正負値）を出力し、mask\_head出力の一部画素が**0または1に飽和**した可能性があります。特に、mask\_headの初期重みが**ランダム**であればseg\_queryとの組み合わせでピクセル毎に微小なランダム差異が出るはずですが、ログ上は完全一様0.5でした。このことから**mask\_headの初期バイアス0/重み0**設定か、あるいは**seg\_query自体がほぼゼロ**だったことが考えられます。後者の場合、1サンプル目学習でseg\_queryに微小な勾配が入り非ゼロになったものの、mask\_headが再初期化されたため**予期せぬ大きなロジット**を生み出した可能性があります。いずれにせよ、**mask\_headを毎回初期化する現在の設計**は、学習の不安定性を高めていると考えられます。

**4. \*\* **勾配スケーリング/精度の問題:** もしモデルをFP16など**混合精度**で学習している場合、勾配のオーバーフロー/アンダーフロー検知や`GradScaler`によるスケーリングが行われていないとNaNが出やすくなります。今回、1サンプル目の最大勾配ノルムは非常に小さい(0.00149)ため勾配爆発ではなさそうですが、**FP16では非常に小さい値はゼロにアンダーフロー**したり、極端に大きい中間値はInfにオーバーフローします。2サンプル目で発生したNaNは、おそらく**損失計算中**に発生したと推測されますが、もし**seg\_query\*\*や他の重みがInf/NaNになっていた場合、その原因はFP16の丸めかもしれません。ただログ上、1サンプル目完了時点ではNaN/Inf検出はなく勾配も正常なので、**FP16単独の問題よりは損失計算の不安定性**と考えるほうが自然でしょう。

以上の分析から、**NaN発生の主原因はフォーカル損失計算の不安定性**であり、**mask\_headの再初期化による学習不能状態**がそれを助長している可能性が高いです。LoRA未適用そのものがNaNの直接原因とは考えにくく、**LoRAは主にメモリ節約や安定したモデル微調整のための手法**で、NaN問題はまず**損失関数と学習方法の修正**で対処すべきです。

## NaN問題への修正方針

**1. フォーカル損失の安定化:** フォーカル損失（またはBCE損失）計算時に**予測確率のクリッピング**を導入します。具体的には、モデルの出力ロジットにシグモイドを適用して確率\$p\$を得る際、`p = clamp(sigmoid(logit), ε, 1-ε)`とし、たとえばε = 1e-6程度の下限・上限を設けます。これにより**log(0)**や**log(1-p)でp=1**といった計算を避けられます。PyTorchには`F.binary_cross_entropy_with_logits(logits, target)`という関数があり、これは内部で数値安定化された形でBCEを計算しますので、**フォーカル損失を計算する際もロジットを直接用いて**一旦BCEを計算し、その結果にフォーカルの重み係数を掛ける方法が堅牢です。例えば以下のように実装します:

```python
import torch.nn.functional as F

# logits: mask_headからの出力 (形状: [N, H*W] 等)
# target: 対応する正解マスク (0/1 のバイナリ値, 形状: [N, H*W])
bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

# フォーカル係数を計算
probs = torch.sigmoid(logits)  # 確率に変換 (ここはFP32で計算)
probs = torch.clamp(probs, min=1e-6, max=1-1e-6)  # log(0)防止のクリッピング
pt = torch.where(target == 1, probs, 1 - probs)   # 正例にはp, 負例には(1-p)
gamma = 2.0  # フォーカル損失のγ（適宜設定）
alpha = 0.25 # フォーカル損失のα（正例重み付け、必要に応じて）
focal_weight = torch.where(target == 1, alpha * (1 - pt) ** gamma,
                                        (1 - alpha) * (pt) ** gamma)
focal_loss = (focal_weight * bce_loss).mean()
```

上記では**確率値のままフォーカルの重みを計算**していますが、こうした実装により`log(0)`などの発生を防げます。また、Hugging Face Transformers等に実装された安定版のBCEロスを活用することで、自前でlog計算をしないようにするのも有効です。重要なのは**εでクリップする**、**FP32で損失計算する**といった工夫で、勾配の爆発やNaNを防止することです。

**2. ダイス損失の安定化:** ダイス係数の実装に**スムージング項**を入れてください。例えば、一般的には

$$
\text{DiceLoss}(P, T) = 1 - \frac{2|P \cap T| + \epsilon}{|P| + |T| + \epsilon}
$$

のように、分母・分子双方に\$\epsilon\$（1e-6など）を加えます。これによって**予測も真値もゼロ**のケースでも \$(0 + \epsilon)/(0 + \epsilon)\$ = 1 となり、DiceLossは0となります（NaNではなくなります）。現在の実装で既にスムージングしているか確認し、不足していれば追加します。またDice損失算出前に**予測マスク値のクランプ**（例えば最低0、最高1に近づきすぎないように）を行うのも安全策になります。

**3. mask\_headの再初期化改善:** 現在各サンプルで`mask_head`を再初期化している設計は、**学習を妨げ不安定化**させている恐れがあります。mask\_headは**SAMデコーダの最終出力層**（おそらく1x1畳み込みや小MLP）と推測されますが、LISAやFoodLMMの手法では**この出力層も含めて学習**させています。したがって、**mask\_headを毎回再構築しない**よう修正し、モデルの一部として固定して持つべきです。具体的には:

* **mask\_headをモデル初期化時に一度だけ構築**し、`model.parameters()`に含めてoptimizerに渡す。
* 各サンプル処理前に行っている`mask_head再初期化`の処理を削除し、**代わりに必要ならforward時にmask\_headの内部状態リセット**（例えば畳み込みのバッファ初期化など）が必要か検討する。通常、単純な線形層や畳み込み層なら状態は重みのみなのでリセット不要です。

この変更により、mask\_headの**重みが蓄積して学習**されるようになります。初期には全出力0.5でも、徐々に出力パターンを変えられるようになり、フォーカル・ダイス損失の勾配が有効に働きます。2サンプル目でのNaNも、mask\_headが学習されていれば極端な出力（全0など）を避け、損失が安定することが期待できます。

**4. 学習率・勾配クリッピング:** 上記修正後もなおNaNが出るようであれば、**学習率を下げる**ことや**Gradient Clipping（勾配ノルムクリップ）**を導入することも有効です。特に大規模モデルでは、ごく少数サンプルで大きく重みを更新すると不安定になりがちです。FoodLMMではLoRA適用時に**初期学習率3e-4、100ステップのウォームアップ**など安定に配慮したスケジューリングを採用しています。また、1ステップ目の勾配ノルムは小さくても2ステップ目に跳ねる可能性もあるため、`torch.nn.utils.clip_grad_norm_(parameters, max_norm)`などで**勾配ノルムの上限**を設けておくと安心です。

**5. 混合精度対応:** もしFP16などで訓練しているなら、**autocast + GradScaler**による自動スケーリングを導入します。具体的にはトレーニングループ内で:

```python
scaler = torch.cuda.amp.GradScaler()
for images, masks in dataloader:
    with torch.cuda.amp.autocast():
        loss = model(images, masks)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

のようにします。これにより、勾配が大きすぎる場合に自動的にscale downされ、NaN発生を予防します。ただし上記の**損失計算の安定化**が根本対策で、GradScalerは保険的役割です。

以上の対策を講じることで、NaN問題は解消する可能性が高いです。特に**フォーカル損失の安定化とmask\_headの学習有効化**が重要ポイントです。

## LoRA/QLoRA導入の検討

**LoRA未導入がNaNの直接原因ではない**とはいえ、プロジェクトの今後（特に最終的にQwen2.5-VL-72Bモデルを扱うこと）を考えると、**LoRAもしくはQLoRAによる微調整**はぜひ導入すべきです。FoodLMMでも、マルチモーダルLLM部分（LLaVAベース）を**LoRAでチューニング**し、SAMデコーダや投影層のみフルチューニングする戦略が採られました。以下に、本プロジェクトに適したLoRA設定と実装指針を示します。

**1. LoRAの適用範囲:** Qwen2.5-VL統合モデルでは、主に**Qwenの言語モデル部分**にLoRAを適用するのが望ましいです。SAM側（特に画像エンコーダ）は既に凍結しており、**SAMデコーダ（mask\_head含む）や画像→言語の投影層**はパラメータ数がそれほど大きくないためフルチューニングで問題ありません。一方、Qwenの言語モデル（数十億〜数百億パラメータ）は、**LoRAによって一部重みに低ランクの微調整を加える**ことでメモリ効率よく学習できます。FoodLMMでは7BのLLaVAにLoRAを適用し、72Bといった大規模モデルにもスケール可能な手法としています。

**2. LoRAハイパーパラメータ設定:** 一般的な設定としては以下が参考になります。

* **LoRAランク (r):** 例えば64や128程度。ランクが高いほど表現力がありますがVRAM消費も増えます。FoodLMMでは具体的数値は公開されていませんが、精細な新能力（セグメンテーショントークン生成など）を学習するため**比較的高めのランク**を用いた可能性があります（例えば**r=64**程度）。開発時3Bモデルではr=32〜64でも充分試せます。72B本番ではメモリを見つつrを決めましょう。
* **LoRAスケーリング (α):** 一般にαはLoRAの効果を調整するスケーリング因子です。元論文ではデフォルトα=rとしていますが、実装によっては**α=2\*r**が用いられる場合もあります。例えばHuggingFaceのPEFTでは`alpha`を指定できますが、特に理由が無ければ**αをrと同値**か**2\*r**に設定します（例: r=64ならα=64か128）。αは大きくしても初期ではLoRA重みがαで1/α倍されてから適用されるため、学習安定性への直接影響は小さいです。
* **LoRAドロップアウト:** LoRA行列に対するドロップアウト率です。過学習防止のため**0.05〜0.1**程度を入れることがあります。FoodLMM実装詳細は不明ですが、一般的には**lora\_dropout=0.05**が初期値として妥当です。
* **対象モジュール:** LoRAを適用するレイヤを指定します。Qwen2.5-VLの言語モデル部分はGPT類似のTransformerブロックで構成され、各ブロック内の**全ての全結合層**に適用するのが効果的です。具体的には**AttentionのQ,K,V投影および出力投影、FFNの各線形層**が対象です。例えばHuggingFace版LLaMAの場合は`q_proj`,`k_proj`,`v_proj`,`o_proj`,`gate_proj`,`up_proj`,`down_proj`といった名前ですが、Qwenモデルのレイヤ名に合わせて指定します。HuggingFaceのtransformers実装でモデルクラスを調べ、`model.named_modules()`からAttentionやFeedForward内のLinear層名称を確認するとよいでしょう。もし名称が把握しづらければ、**PEFTのLoraConfigで`target_modules=["Wq", "Wk", ...]`のように部分一致指定**も可能です。
* **LoRA実装:** Hugging FaceのPEFTライブラリを使うと容易です。例として:

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA設定
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # Qwenは自回帰言語モデル
    inference_mode=False,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "down_proj", "up_proj", "gate_proj"]  # Qwenの実際のLinear層名に合わせること
)
# ベースモデルにLoRA適用
model = get_peft_model(model, lora_config)
print("LoRAモデルパラメータ数:", sum(p.numel() for p in model.parameters() if p.requires_grad))
```

上記では`target_modules`に典型的なTransformer内部の線形層名をリストしています。**Qwenの命名規則に合わせて**必要なら修正してください。PEFTは名前部分一致で対象モジュールを探すので、例えば全Attention層に適用したければ`"attn"`など共通部分でも指定可能です。適用後、`model`はLoRA付きモデルとなり、`model.print_trainable_parameters()`で凍結/学習パラメータ数を確認できます。

**3. QLoRAの活用:** 将来的に72Bモデルを扱う際は、**QLoRA**（Quantized LoRA）でさらなるメモリ削減を図ります。QLoRAでは**ベースモデル重みを4bit量子化**しつつ、LoRAで微調整を行います。具体的には、`bitsandbytes`ライブラリでモデルをロードする際に`load_in_4bit=True`とし、quantization configでNF4量子化（norm付き4bit）を指定します。その上で上記と同様の`get_peft_model`適用が可能です。PEFTの`LoraConfig`はモデルが量子化されていても動作します。注意点は、Optimizerに`bitsandbytes`のAdamW (8-bit Adamなど)を用いてメモリ節約・安定化する点です。HuggingFaceのtransformersにはQLoRA用ユーティリティ（例: `BitsAndBytesConfig`）がありますので、

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # ダブル量子化
    bnb_4bit_quant_type="nf4",        # NF4量子化
    bnb_4bit_compute_dtype=torch.float16  # またはtorch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", quantization_config=bnb_config, device_map="auto")
```

のようにモデル読込時に指定します。その後LoRA適用し、Optimizerは:

```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)
```

等とすると良いでしょう。QLoRAでは**微調整可能なパラメータがLoRA層のみ**となるため、72Bクラスの巨大モデルでもGPUメモリ数十GBで学習が可能になります。FoodLMM論文でもLoRAでのチューニングにより巨大モデルの学習を実現した旨が示唆されています。

**4. LoRA導入による副次効果:** LoRAを使うことで、**ベースモデル（Qwen）の元の知識を保ちつつ新タスクへの適応**が期待できます。FoodLMMでは、**特殊トークン（<SEG>等）の生成能力**やテキスト応答の調整をLoRAで学習させています。本プロジェクトでも、LoRAによって\*\*<SEG>や\[REJ]トークンの適切な使用方法**を学習し、モデルがテキストとマスク出力を連携して生成できるようになります。さらに、将来的に72Bモデルに移行する際もLoRAなら**既存の微調整結果をマージ\*\*して引き継ぐことができます（PEFTの`merge_and_unload`や独自スクリプトでLoRA重みを合成可能です）。

**5. LoRA適用後の学習戦略:** LoRAを導入したら、学習スクリプト上では**Optimizerに渡すのはLoRA層（および他の学習対象層）のみ**になります。PEFTの`get_peft_model`を使った場合、自動的に`.parameters()`でLoRA層も含まれますが、内部でベースモデルの勾配は止めてくれています。`model.print_trainable_parameters()`で確認すると、「学習可能パラメータ: ～」という出力が得られるはずです。FoodLMMではLoRAでLLaVAを調整しつつ、**SAMデコーダや投影層も学習可能にした**と述べています。本プロジェクトでも、**Qwen本体はLoRA**, **SAMデコーダ（mask\_head含む）と投影層は全パラメータ直接学習**という構成が適切でしょう。この場合、Optimizerに渡す`model.parameters()`はLoRAもデコーダも全て含むため、特別な処理は不要ですが、**学習率をパラメータグループで分ける**ことは検討できます。例えば、**言語モデルLoRA部分は低め学習率**（例: 2e-4）、**視覚デコーダ部分はやや高め**（例: 1e-3）とすることで、微調整幅の差を吸収できます。PEFTでは各層名でパラメータフィルタ可能なので、optimizer定義時に名前条件でグループ化すると良いでしょう。

## スクリプト修正内容のまとめ

最後に、以上の指針を具体的に`train_minimal.py`に反映する方法をまとめます。

* **損失計算の修正:** `train_minimal.py`内でフォワード時に計算しているフォーカル損失・ダイス損失の実装を見直します。自前実装の場合、上記のように`binary_cross_entropy_with_logits`を用いた安全な計算に置き換え、`logits -> sigmoid -> log`の明示的計算を避けます。ダイス損失には分母分子に`+1e-6`を追加します。例えば:

  ```python
  # 予測マスクlogits取得後
  bce_loss_map = F.binary_cross_entropy_with_logits(pred_mask_logits, target_mask, reduction='none')
  prob = torch.sigmoid(pred_mask_logits)
  prob = torch.clamp(prob, 1e-6, 1-1e-6)
  pt = torch.where(target_mask==1, prob, 1-prob)
  focal_factor = torch.where(target_mask==1,
                             alpha * (1-pt)**gamma,
                             (1-alpha) * (pt**gamma))
  focal_loss = (focal_factor * bce_loss_map).mean()

  # ダイス損失計算（バイナリマスク想定）
  pred_mask = (prob > 0.5).float()  # 0-1に二値化するか、あるいは確率のまま計算
  intersection = (pred_mask * target_mask).sum()
  dice_coeff = (2 * intersection + 1e-6) / (pred_mask.sum() + target_mask.sum() + 1e-6)
  dice_loss = 1 - dice_coeff
  ```

  などとします（二値化せずそのまま確率値でダイス計算する手法もありますが上記はシンプルな形です）。計算した`focal_loss`と`dice_loss`を足した`total_loss`を**最終的なセグメンテーション損失**とします。なお、上記では`alpha`や`gamma`は適宜設定（例えばα=0.25, γ=2）してください。また既存コードでフォーカル損失を**クロスエントロピーそのものではなく**`(p_t)^γ`形式で実装している場合は特に注意し、**不安定な部分のみクリップ/安定化**するよう調整します。

* **mask\_head初期化ロジックの変更:** `train_minimal.py`冒頭でモデルを構築する際、mask\_head（SAMのマスク予測層）を**統合モデルの一部として宣言**します。もし現在`forward`内で`mask_head = MaskHead()`のように都度生成しているなら、それを**モデルの`__init__`で一回だけ生成**しインスタンス変数に保持します。例えば:

  ```python
  class IntegratedSAMQwen(nn.Module):
      def __init__(...):
          super().__init__()
          ...
          self.mask_head = MaskHead(...)  # 一度だけ生成
      def forward(self, image, ...):
          ... 
          mask_logits = self.mask_head(seg_query, image_embedding)
          ...
          return mask_logits
  ```

  とし、`train_minimal.py`で毎サンプル前に呼んでいた`mask_head再初期化`は削除します。**重要**: mask\_headが内部でバッチごとに異なる動作を期待している場合（例えばSAMは複数のマスク提案を出す機構があります）、その**可変部分はパラメータではなく入力として扱う**べきです。具体的には、mask\_headに**mask\_embeddings**や**iou\_prediction**などがある場合、それらを毎回再初期化せず**学習時には固定**または**使わない**ようにします。LISAでは「embedding-as-mask」方式で、単一の<SEG>トークン埋め込みをマスク出力に転換しています。SAMのデコーダ部分は**Transformer＋出力層**なので、通常学習可能層にして問題ありません。mask\_head固定化後は**optimizerにmask\_head.parameters()が含まれる**ことを確認しましょう（ログに出ている学習可能パラメータ数も増えるはずです）。

* **LoRAの組み込み:** 上記のようにPEFTを用いてモデルラッパーを作成するか、もしくは既存コードで独自にLoRAを差し込んでも構いません。よりシンプルには、HuggingFaceモデルをLoRA化するツールを利用できますが、ここではコードベースでの組み込みを示します。

  1. **ライブラリ導入:** `peft`と`bitsandbytes`（後者はQLoRA用）を`requirements.txt`に追加します。またtransformersのバージョンがPEFT対応済みか確認してください（少なくともv4.30+推奨）。

  2. **モデル読み込み:** 3BモデルであればFP16でも載ると思いますが、72B見据えて**QLoRAのオプション**を準備しておくと良いでしょう。環境に合わせ、3Bでは通常のFP16+LoRA、72Bでは4bit+LoRAという風に条件分岐してもOKです。例:

     ```python
     use_q4 = (model_name_or_size == "Qwen/Qwen2.5-VL-72B-Instruct")
     if use_q4:
         bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)
         base_model = AutoModelForCausalLM.from_pretrained(model_name_or_size,
                                                           quantization_config=bnb_config,
                                                           device_map="auto")
     else:
         base_model = AutoModelForCausalLM.from_pretrained(model_name_or_size,
                                                           device_map="auto", torch_dtype=torch.float16)
     ```

     Qwen2.5-VLの場合、`AutoModelForCausalLM`ではなく専用の`AutoModelForVisionLanguage`があるかもしれません（HuggingFaceモデルカードを確認ください）。その場合も内部構造は似ているのでLoRA適用法は変わりません。

  3. **LoRA適用:** 上記で述べた`LoraConfig`を用いてモデルをラップします。例えば:

     ```python
     from peft import LoraConfig, get_peft_model, TaskType
     lora_config = LoraConfig(
         task_type=TaskType.CAUSAL_LM,
         r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj","down_proj","up_proj","gate_proj"],
         lora_dropout=0.05
     )
     model = get_peft_model(base_model, lora_config)
     model.print_trainable_parameters()
     ```

     これにより、Qwenモデルの該当線形層に低ランクの追加重みが挿入されます。`print_trainable_parameters()`の出力で、`trainable params: X || all params: Y || X/Y (%)`と表示されますので、**学習対象パラメータが劇的に削減**されていることを確認できます。LoRA適用後も、既存のSAM統合部分（SAMデコーダやmask\_head、投影層）は`model`内にそのまま存在するはずです。もし`get_peft_model`がうまく統合モデルに適用できない場合は、Qwen部分だけLoRA化してから統合する必要があるかもしれません。その際は統合モデルクラスの中でQwen部分にLoRAを適用する工夫が要りますが、理想的には**統合後のモデル全体**に対し一度`get_peft_model`を呼べれば簡単です。

  4. **Optimizer設定:** LoRA導入後は、基本的に`model.parameters()`で得られるパラメータには\*\*LoRA層（学習可）**と**凍結されたベース層（学習不可）\*\*が含まれます。したがって`optimizer = AdamW(model.parameters(), lr=...)`で問題ありません。内部的に学習不可パラメータは勾配が0になるだけなのでoptimizerに載せても害はありません（気になる場合はfilterで`p.requires_grad`なパラメータのみにしてもOKです）。前述のように、視覚デコーダ系とLoRA層で学習率を変えたければ、例えば:

     ```python
     lora_params = [p for n,p in model.named_parameters() if "lora" in n and p.requires_grad]
     other_params = [p for n,p in model.named_parameters() if "sam_decoder" in n and p.requires_grad]  # 名前は適宜
     optimizer = AdamW([
         {"params": lora_params, "lr": 2e-4},
         {"params": other_params, "lr": 1e-3}
     ], weight_decay=0.0)
     ```

     のようにパラメータグループを組みます。FoodLMMではweight decay=0、AdamW使用、ウォームアップ有りでしたので、それにならい**正則化やoptimizer設定**も行います。

  5. **学習スクリプト上の変更:** LoRA適用後も`train_minimal.py`のエポック・バッチ処理の流れ自体は同じです。ただし**巨大モデルを想定すると`GradientAccumulation`や`分散学習(DeepSpeed)`**が必要になるかもしれません。まずは単一GPUで回せる3BモデルでNaNが出ず学習が進むか確認しましょう。その際、出力ログに**各エポックでlossが徐々に低下**していくこと、及び**学習後の推論でそれなりにマスクが出力**されることを検証します。ダミーデータとはいえ、例えば常に背景のみの画像を与えているなら学習は進まないので、ダミーデータも適度に難易度を持たせます（NaN回避のため **真値マスクが全ゼロのケースは避ける** こともポイントです）。

以上の変更を加えた後、再度`train_minimal.py`を実行してみてください。修正が適切であれば、**2サンプル目でのNaNは発生せず**、lossが徐々に減少していくはずです。またLoRAを組み込んだことで、学習可能パラメータ数が減りメモリ効率も向上します。最終的には、この基盤モデル(LISA改)を用いてFoodLMM改をファインチューニングし、**画像中の料理や食材の量（領域）を高精度に推定できる**ことを目指します。各段階で得られた知見を踏まえ、より良い安定性と性能を追求してください。

**参考文献:**

* LISA手法（Language Instructed Segmentation Assistant）の概要と学習戦略
* FoodLMMにおける特殊トークンとセグメンテーション統合、およびLoRA微調整の適用
* LoRA/QLoRAの一般的なベストプラクティス
