承知しました。これから、現行コードベース（Qwen2.5-VL-3B + SAM2.1）をもとに、Qwen ViT主体構成へ完全に移行し、視覚入力→Qwen ViT→SAMマスクデコーダという流れに切り替えるための詳細な設計・修正指示書（コード含む）を作成します。

LoRA適用構成や学習スクリプト（train\_minimal\_lora.py）は維持しつつ、mask\_headや特殊トークン設計も必要に応じて改変可能な前提で、最適化設計をリサーチして構成案を提示します。少々お待ちください。


# Option A 移行の詳細設計および実装指示

## 現行実装の概要と課題

現行のLISA改モデルでは、**SAM2.1の画像エンコーダ**を用いて画像特徴を抽出し、それを**VisualProjector（Q-Former）**で変換してQwen2.5-VLのLLM部分に入力しています。また同じ画像特徴をSAM2.1の**マスクデコーダ**にも入力し、セグメンテーションマスクを出力する構成です。つまり、**画像→SAMエンコーダ→特徴**を共有し、**LLM**と**セグメンテーション**の双方に活用する統合フローになっています。

この実装により、画像エンコードの計算は1回で済み\*\*（テキスト生成とセグメンテーションを同時実行可能）**という利点があります。しかし、SAM2.1の画像エンコーダは**大規模**であり、将来的にドメイン特化した精密な量推定を行うには**最新VLMであるQwenの視覚能力\*\*を直接活用するほうが望ましいと考えられます。また、現行はQwen側の画像エンコーダを利用しておらず、Qwen本来の視覚‐言語統合能力を十分に発揮していない可能性があります。

以上を踏まえ、**Option A**では画像エンコーダをSAMから**Qwen2.5-VLのViT**に切り替え、より洗練された視覚表現を活用します。この変更に伴い、SAMマスクデコーダへの接続方法やマスク出力方法も最適化する必要があります。

## Option A の設計方針

Option Aへの移行にあたり、以下の設計指針に従ってモデルを再設計します。

### 1. 画像→Qwen ViT→SAMマスクデコーダへの切り替え

**画像エンコーダはQwen2.5-VL側のViT**（視覚エンコーダ）を使用し、**SAM2.1独自のViT**は使用しません。すなわち、**画像入力→Qwenの視覚バックボーン→特徴マップ→SAMマスクデコーダ**というパイプラインに完全に切り替えます。これにより、Qwenの高度な視覚特徴抽出能力を直接活かしつつ、SAMの優れたマスク生成機構を統合します。

実装上は、**現行のモンキーパッチを反転**させるイメージです。現在は「Qwenの視覚エンコーダをSAM2.1に置換」していましたが、Option Aでは**SAMの画像エンコーダ処理を除去**し、**Qwenの視覚エンコーダを主導**にします。QwenのViTが出力する特徴マップを**そのままSAMマスクデコーダに入力**できるよう変換・調整します。これにより、画像エンコードはQwen側で1回だけ行い、その出力をテキスト生成とセグメンテーションの両方で使い回す設計となります。

### 2. 複数マスク出力の設計

**複数物体のセグメンテーション出力**については、**SAM2.1のマスクデコーダ設計**（複数のマスクトークンによるマスク提案）を基本的に踏襲します。現行実装で用いているmask\_headや特殊トークンの枠組みは、Option Aでも**基本方針として維持**しつつ、Qwenの特徴マップに適合するよう**再設計**します。

具体的には、SAMマスクデコーダ内部で使用される**マスクトークン**（通常3つのマスク出力に対応）を引き続き活用し、**最大3つ程度のマスク**を同時推論可能な構造を保ちます。これにより、1つの入力画像に対して**複数の候補領域**をセグメント化し、後段で適切なマスクを選択・利用できます。必要であればマスクトークン数やデコーダの構造を**可変に拡張**することも検討可能ですが、まずはSAM既定の**固定数トークン**による安定した出力方式を採用します。

なお、現行で試みていた**特殊トークンを用いた設計変更**（もし存在する場合）は、Option Aでは**最適解を優先**するため大幅な変更も許容します。ただし、まずは**既存SAM方式に沿った実装**で堅実に動作検証し、その後必要に応じて拡張する方針とします。

### 3. LoRA適用と量子化設定の継承

モデルの学習・ファインチューニングに関しては、現行の`train_minimal_lora.py`で確立した**LoRA適用**手法および**量子化（4bit/AWQなど）設定**を**維持**します。Option Aへのアーキテクチャ変更後も、**低VRAM環境で大規模モデルを効率良く学習**できる現在の仕組みをそのまま活用します。

具体的には、Qwen2.5-VL-72Bなど大規模LLM部分には引き続きLoRAを適用し（必要に応じて視覚エンコーダやマスクデコーダにもLoRAを適用することを検討）、学習時の微調整可能パラメータを絞ります。同時に、モデル読み込み時には**事前に量子化（例: `load_in_4bit=True` や AWQによるweight量子化）**を行う設定を保持し、メモリ消費を抑えます。現行のスクリプトに実装されているこれら設定はそのままOption Aでも**互換性がある**ため、ベースラインの学習方法論は変更不要です。

## 実装変更の詳細

以上の設計方針に基づき、コードベースに対し以下の具体的な変更を行います。修正は主に**モデル構築部分**（画像エンコーダ差し替え、特徴変換）、**マスクデコーダ入出力の調整**、および**周辺設定**に及びます。各変更箇所について、可能な限り曖昧さのない指示とコード例を示します。

### 1. Qwen視覚エンコーダへの切替実装

**対象ファイル:** `model/sam_qwen_model.py`（統合モデル実装）および必要に応じて`model/visual_projector.py`（視覚特徴プロジェクタ）。

* **Qwenモデルの読込:** Option AではQwen2.5-VLのビジュアルエンコーダを使用するため、モデル初期化時に**Qwenのマルチモーダルモデル**を読み込みます。例えばHugging Faceのオートローダを使って `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B", ...)` あるいは公式実装経由でモデルをロードし、`self.qwen_model`として保持します（7B/3Bは動作確認用途、最終的には72Bを想定）。既存コードでSAMエンコーダやQwen LLMをロードしていた部分を見直し、**Qwenモデル全体を一度に読み込む**か、必要部分（ViT+Q-FormerとLLM）を個別にロードする方法に変更します。

* **SAMエンコーダの除去:** 現行の`SamQwenModel`初期化で行っている**SAM2.1画像エンコーダのセットアップ**（およびQwen視覚エンコーダとのモンキーパッチ処理）を削除します。例えば、`self.sam_image_encoder = ...` や Qwen側`vision_tower`の差し替え等のコード行を撤廃します。Option AではSAMから画像エンコーダは使わないため、SAM関連では**マスクデコーダとプロンプトエンコーダのみ**を利用する形になります。SAMの重量モデルをロード・保持する必要も無くなるため、その分メモリ節約にもなります。

* **Qwenエンコーダの適用:** `forward`メソッド等で画像を処理する際は、**Qwenの視覚エンコーダ**を使用します。具体的には、入力画像（tensor）をQwenモデルのビジョンタワーに通し\*\*特徴マップ（パッチ埋め込みの集合）\*\*を取得します。例えば、Qwenモデル内の視覚エンコーダを呼び出すインターフェイスが用意されている場合：

  ```python
  # Pseudo-code: Qwenモデルの視覚エンコーダ部分を呼び出して特徴取得
  vision_tower = self.qwen_model.vision_tower[0]  # 仮: Qwenの視覚モデル（ViT）
  image_embeds = vision_tower(image_tensor)       # 特徴マップ取得 (形状: [B, N, C])
  ```

  Qwenの実装によって実際の呼び出し方法は異なりますが、多くの公開モデルでは上記のように`.vision_tower`や類似属性からViTモジュールにアクセスできます。**重要**なのは、この`image_embeds`として**画像のパッチ特徴列**（バッチサイズB、パッチ数N、次元C）を取得することです。これが**SAMマスクデコーダへの入力**および**Qwen内部Q-Formerへの入力**の共通出発点となります。

* **特徴マップの形状整形:** QwenのViTから得られた`image_embeds`（\[B, N, C]）を**空間的な特徴マップ**に整形します。例えば、画像を正方形にリサイズ入力し、ViTのパッチ埋め込み数NがH×Wとなるように調整します（H≒W）。典型的には、QwenのViTは16x16や14x14のパッチサイズを持つ可能性が高いです。**入力画像サイズ**を適切に設定し（例: 1024×1024にリサイズ）、**Nが正方格子**になるようにします（例えばパッチサイズ16なら1024÷16=64で**64×64 = 4096**パッチ）。この場合、`image_embeds`の形状を\*\*(B, 4096, C)**から**(B, C, 64, 64)\*\*にリシェイプできます。

  コード上は以下のような処理を追加します（CはQwen視覚特徴次元、64は仮定したパッチグリッドサイズ）:

  ```python
  B, N, C = image_embeds.shape
  H = W = int(N ** 0.5)  # Nが正方に近い前提
  image_feat_map = image_embeds.permute(0, 2, 1).reshape(B, C, H, W)
  ```

  ※Nが完全な平方数にならない場合（例: パッチサイズが画像サイズに合わない場合）は、**Padding**や**最近傍の平方数までのトリミング**で調整します。または**ViT側で出力の2次元形状(H, W)を取得**できる関数があればそれを利用します。Option Aの実装では、画像前処理で想定どおりN=H×Wとなるようにするのが確実です。

* **VisualProjectorの役割変更:** 現行の`VisualProjector`は、SAMエンコーダ出力をQwenのLLM用特徴量にプロジェクションするために**Q-Former**あるいは単純な線形層を提供していました。Option Aでは**Qwen側のQ-Formerを活用**できるため、VisualProjectorの処理内容を見直します。具体的には、

  * **USE\_QFORMER=Trueの場合:** Qwenモデルに内蔵のQ-Former（学習済みの視覚クエリトークンによる変換機構）をそのまま使用します。すなわち、上で得た`image_embeds`を使って**QwenのQ-Formerモジュール**からLLM入力用のクエリ特徴を取得します。多くの場合、Q-FormerはQwenモデル内で`vision_tower`に続いて自動的に適用されます。モンキーパッチを解除したことでQwenモデル本来の挙動（ViTの後にQ-Formerで32個程度のクエリベクトル抽出）が復活するため、**VisualProjectorを明示的に呼ばず**とも、Qwenモデルに画像を渡せば内部で完結するはずです。
  * **USE\_QFORMER=False（Linear使用）の場合:** QwenのQ-Formerを使わず独自に特徴射影する設定であれば、`VisualProjector`を**単純な線形層**として利用し、`image_embeds`を平均プーリングするなどしてLLM入力ベクトルに変換します。ただし、基本はQ-Formerを活かす設計とし、Linearモードはデバッグや簡易テスト用に残す程度で構いません。

  実装上は、`VisualProjector.forward()`内をOption A用に修正します。例えば擬似コード:

  ```python
  class VisualProjector(nn.Module):
      def __init__(self, use_qformer=True, ...):
          ...
      def forward(self, image_embeds):
          if self.use_qformer:
              # QwenモデルのQ-Formerを使用してクエリ取得
              queries = self.qwen_model.q_former(image_embeds)  # 仮想コード
              return queries  # 形式はLLMが受け取れるよう調整
          else:
              # 平均プーリング＋線形射影
              pooled = image_embeds.mean(dim=1)  # [B, C] 平均特徴
              return self.linear(pooled)         # [B, LLM_input_dim]
  ```

  ※ 実際のQwenモデルでは`q_former`モジュールへの直接アクセスが必要になります。Qwenの公式実装・HFモデルの構造に合わせて適宜修正してください。場合によっては、**Qwenモデルに画像と言語を一緒に与えてforward**し、LLM内部から視覚クエリ出力をフックする実装となるかもしれません。この点は後述する**LLMテキスト生成**部分でも触れます。

### 2. SAMマスクデコーダとの統合

**対象ファイル:** `model/sam_qwen_model.py`（forward処理）, 必要に応じて`sam2`フォルダ内のSAM関連クラス（マスクデコーダ、プロンプトエンコーダ等）。

* **SAMマスクデコーダの初期化:** 現行ではSAM2.1のモデル全体、または一部を読み込んでいましたが、Option Aでは**マスクデコーダ部分のみ**を利用します。Facebook Research提供のSAM2.1実装（おそらく`sam2`ディレクトリ配下）から**MaskDecoderクラス**等を直接インスタンス化するか、既存のSAMモデルインスタンスからマスクデコーダ部分を切り出して`self.mask_decoder`として保持します。例えば:

  ```python
  from sam2.modeling.mask_decoder import MaskDecoder
  self.mask_decoder = MaskDecoder(...必要なパラメータ...)
  ```

  **注意:** MaskDecoderの初期化には、**画像埋め込みのチャネル数**や**マスクトークン数**、**transformerの構成**など、SAMエンコーダに依存するパラメータが必要です。Option AではSAMエンコーダを使わないため、**これらパラメータをQwenのエンコーダ出力に合わせて設定**する必要があります。例えば、Qwen ViT出力のチャネル次元Cに合わせてMaskDecoderの期待する`encoder_dim`を設定します（もしくはQwen特徴を別途プロジェクションして既存値に合わせる）。マスクトークン数やtransformer層数はSAM2.1のデフォルト（mask\_token 3個など）をそのまま用います。

* **画像特徴の投入:** 前ステップで取得した`image_feat_map`（形状\[B, C, H, W]）をMaskDecoderに与えます。MaskDecoderの`forward`呼び出しには通常、**画像エンコーダの出力特徴**に加え、**対応する位置エンコーディング**や**ポイントプロンプトの埋め込み**、**マスクトークン**などを渡します。Option Aでもこれらを適切に準備してから呼び出します。

  * **位置エンコーディング:** SAMエンコーダでは画像サイズに応じたpositional embeddingがありますが、Qwenの特徴マップに対しては**新たに位置エンコーディングを用意**する必要があります。簡易な方法として、MaskDecoder内部で使用している**学習済みの埋め込み**をそのまま流用することが考えられます。SAM2.1のエンコーダ出力と我々の`image_feat_map`の空間次元(H, W)が一致するなら、SAM提供の位置埋め込みテンソル（例えば正弦波PEや学習PE）を取得して利用できます。もし一致しない場合は、**新規に2次元Positional Embeddingを定義**し、MaskDecoderとともに学習させます。
    コード上、例えばMaskDecoder呼び出しが

    ```python
    mask_logits, iou_pred = self.mask_decoder(image_feat_map, image_pe, sparse_prompt_emb, dense_prompt_emb)
    ```

    というシグネチャの場合、それに対応する`image_pe`（画像位置エンコーディング）を準備します。SAMエンコーダ由来のPEが使えるならそれをロードしリサイズ、あるいは`torch.linspace`等で正弦波PEを生成しても構いません。\*\*Option A初期実装ではまず単純な位置エンコーディング（例えばサインカーブ埋め込み）\*\*で試し、精度に応じて後から精密なものに置き換える戦略でもよいでしょう。

  * **プロンプト埋め込み:** 入力にポイントやバウンディングボックス等のプロンプト情報（ラベル付き点など）がある場合、それをエンコードする処理は**現行のまま維持**します。おそらくSAM2.1の**PromptEncoder**クラスを用いて、点座標やボックス座標を埋め込みベクトル（`sparse_prompt_emb`）および\*\*デンスな手法があれば`dense_prompt_emb`\*\*に変換しているはずです。Option Aでも、SAMのPromptEncoderを利用して入力プロンプトを埋め込み、MaskDecoderに渡します。この部分のコード変更は不要ですが、**PromptEncoderの初期化**にSAMエンコーダの出力尺度が関係している場合は確認が必要です（例えばポイント座標正規化に画像サイズを用いる等）。基本的に1024x1024基準で正規化する処理であれば、Option Aでも画像を同サイズにしていれば問題ありません。

  * **マスクトークン:** MaskDecoderには**学習済みのマスクトークン（query token）**があり、通常は3つ（デフォルト）+1個（no-maskトークン）で初期化されます。Option AでもMaskDecoder内に既定通り用意されたものを使います。特に変更を加えなくても、MaskDecoder.forward時に自動的にmask\_tokensが処理に含まれる実装になっている可能性があります（具体的にはMaskDecoder内部でmask\_tokensがembeddingとして登録され、Transformerに投入される）。もしMaskDecoderの使用方法として**明示的にmask\_tokensを渡す必要がある**場合は、初期化時に取り出しておき`forward`呼び出しに含めます。例:

    ```python
    mask_tokens = self.mask_decoder.mask_tokens  # (num_mask_tokens, C)
    mask_logits, iou_pred = self.mask_decoder(image_feat_map, image_pe, sparse_prompt_emb, dense_prompt_emb, mask_tokens)
    ```

    Option Aではデフォルト3個のmask\_tokensで十分ですが、将来的に必要に応じて`self.mask_decoder.num_mask_tokens`を増やすことでマスク提案数を拡張できます。

* **マスク出力の取得:** MaskDecoderの出力は**マスクのロジット**（通常は低解像度：例えば256×256）および**各マスクのIoU予測値**です。現行実装では`results['mask']`として最終マスク（または複数マスク）を取得していたかと思われます。Option Aでも、MaskDecoderの出力`mask_logits`を**アップサンプリング**して最終マスクを得ます。SAMでは**画像エンコーダの高解像度出力**を用いてマスクロジットをアップサンプル（256→1024サイズ）する仕組みがありますが、今回は画像エンコーダが無いので**簡易に双線形補間**で元の画像サイズまで拡大してもよいでしょう。もしSAM2.1の実装内に**アップサンプラ**（例えば`sam2.modeling.mask_decoder.MaskDecoder.output_upscaling`のようなモジュール）が含まれていれば、それも一緒に利用可能です。

  例として、シンプルにPyTorch関数でアップサンプルする場合:

  ```python
  mask_lowres = mask_logits.squeeze(1)          # (B, 256, 256) 1チャネルマスク
  mask_up = F.interpolate(mask_lowres, size=(orig_h, orig_w), mode='bilinear')
  final_mask = (mask_up > 0).float()            # 閾値0で二値化 or そのまま連続値マップでも可
  results['mask'] = final_mask
  ```

  ここで`orig_h, orig_w`は入力画像の実寸サイズです。Option A初期段階ではまず**256^2→1024^2への単純補間**で十分ですが、可能であれば**SAM2.1の学習済みアップサンプリング層**（あるいは関連する出力層ウェイト）を活用すると精度向上が見込めます。

### 3. テキスト生成（LLM）部分の調整

**対象ファイル:** `model/sam_qwen_model.py`（forward内でのLLM呼び出し）, `train_minimal_lora.py`（テキスト生成の損失計算）。

* **画像特徴のLLM入力:** 前述の通り、Qwenモデルには視覚特徴をLLMに取り込む**Q-Former機構**が内蔵されています。Option Aではモンキーパッチを解除したことで、**Qwenモデル本来のforwardを活用**できます。例えば、テキスト生成質問`question`が与えられた場合に:

  ```python
  # Qwenモデルに画像テンソルと質問テキストを与えてテキスト生成
  output = self.qwen_model.generate(**{ "images": image_tensor, "texts": [question], ... })
  generated_text = output[0]
  ```

  のように、Qwen提供のAPIで画像+テキストから直接回答を得ることが可能です（実際のインターフェースはモデル実装によりますが、Qwen2.5-VLのドキュメントに従ってください）。この場合、**内部でViT→Q-Former→LLMが連携**し、テキストが生成されます。

* **重複計算の回避:** 上記のように`self.qwen_model.generate`等を呼ぶと、内部で**再度画像エンコードが行われる**可能性があります。既にMaskDecoder用に一度`vision_tower`で特徴を計算しているため、**二重計算の回避**が望まれます。これには以下のアプローチがあります:

  1. **Qwen内部処理へのフック:** Qwenモデルのコードを解析し、画像エンコーダ→Q-Former→LLMの処理手順を把握します。例えば、`self.qwen_model`内に`forward(image_embeds, input_ids, ...)`のようなメソッドがあり、そこで`vision_tower(image)`→`q_former(vision_output)`→`transformer(decoder_input)`としていれば、途中結果を差し込むことができます。具体的には、一度自前で`vision_output = vision_tower(image)`を計算し保存、次に`self.qwen_model`を呼ぶ際に内部で再計算しないよう**vision\_tower部分をモンキーパッチ**します（現在のモンキーパッチとは逆方向のパッチですが、同様の手法で可能です）。例えば疑似コード:

     ```python
     vision_output = vision_tower(image_tensor)  # 自前で画像エンコード
     # Qwenモデルのvision_tower.forwardを一時的に差し替え
     def _patched_vision_forward(*args, **kwargs):
         return vision_output
     original_forward = vision_tower.forward
     vision_tower.forward = _patched_vision_forward
     # Qwenモデルでテキスト生成（内部でvision_tower.forwardが呼ばれるが差し替わっている）
     generated_text = self.qwen_model.generate(question, use_image=True)
     # 後片付け
     vision_tower.forward = original_forward
     ```

     上記により、Qwenモデルは内部でvision\_towerを呼び出しても既に計算済みの`vision_output`を返し、無駄な演算を避けられます。実装時はQwenモデルの構造に合わせ、正確にパッチを当ててください。
  2. **Qwenモデルのモジュール再利用:** もう一つの方法は、**QwenのQ-FormerとLLMモジュールを直接呼び出す**ことです。例えば`self.qwen_model.q_former`や`self.qwen_model.language_model`（仮称）といった内部モジュールを順次適用し、最終的にテキストを生成します。この場合、LLM部分は通常**トークナイザでquestionをID化→LLMデコーダforward**→生成といった手順になるため実装コストが高くなります。基本は前者（generateのパッチ）で対応し、うまくいかない場合に内部実装に踏み込む形が良いでしょう。

* **出力形式の統一:** 現行`forward`では戻り値`results`に`'generated_text'`キーでLLM出力文字列を、`'mask'`キーでマスクテンソルを返していたはずです。Option Aでもこのインターフェースを維持します。つまり、上で取得した`generated_text`（文字列）を`results['generated_text']`に格納し、MaskDecoderから得た`final_mask`（テンソル）を`results['mask']`に格納して、まとめて返します。これにより、**テキスト+マスクの統合推論**APIが従来通り動作し、`test_integrated_sam_qwen.py`等のテストコードも最小限の修正で済みます。

* **ロス計算への反映:** `train_minimal_lora.py`では、おそらく\*\*言語生成損失（例: 次単語予測のクロスエントロピー）**と**マスク予測損失（例: IoU損失やDice損失）\*\*を計算し、和または重み付き和で最終損失としていると推測します。Option Aへの変更後も、**損失計算のロジック自体は同じ**で構いません。ただし、**出力テンソルの形状や型**に変更がないか確認してください。例えばマスク出力がfloat型の\[0,1]値になっている場合、損失計算側でシグモイドをかけていたなら、Option Aでも同様に処理する必要があります（もしくはMaskDecoderからのロジットに対し`BCEWithLogitsLoss`を使う等設計によります）。実装後、**ダミーデータセット**で`train_minimal_lora.py`を動かし、損失が適切に降下する（学習が進む）ことを確認してください。

### 4. LoRAおよび量子化設定の引継ぎ

**対象ファイル:** `train_minimal_lora.py`, `config_qwen_sam.py` など（LoRAラッパーの適用部分、モデル読み込み部分）。

* **LoRA対象の確認:** Option Aではモデル構造が変わるため、LoRAを適用するモジュール一覧を確認・更新します。現行ではおそらく**QwenのLLM層**（TransformerのW\_qkvやW\_oなど）にLoRAを刺していたと思われます。Option Aでも基本方針は同じですが、**Qwenの視覚エンコーダやSAMマスクデコーダも学習させるか**を検討します。初期段階では、巨大なLLM部分にLoRAを適用し、**視覚エンコーダ（ViT+Q-Former）は凍結**、**マスクデコーダも凍結**で十分でしょう。これは、食品画像の量推定タスクでは主に言語モデル部分（応答の表現や推論論理）を適応させる狙いがあるためです。しかし、もしセグメンテーション精度自体を向上させる必要が出てくれば、**マスクデコーダ内のTransformerブロックや出力層**にもLoRAを適用することで微調整可能です。実装的には、PEFTライブラリの`peft.LoraModel`に渡す`target_modules`リストに、例えば `"qwen_model.transformer.h.*.mlp"`（LLM部分）に加え、`"mask_decoder.transformer.blocks.*.attn"` など該当層を追記するイメージです（実際の層名はモデル定義に合わせてください）。

* **量子化設定の維持:** `config_qwen_sam.py`等で定義されている`TORCH_DTYPE`（float16等）や`load_in_4bit`オプション、またはAWQに関する設定はそのまま使用します。Qwen2.5-VL-72Bを扱う際、**4bit量子化+LoRA**は実運用上不可欠な組合せですので、Option Aでも引き続き適用します。モデル読み込み部分で例えば:

  ```python
  self.qwen_model = AutoModelForCausalLM.from_pretrained(
      QWEN_MODEL_NAME,
      device_map="auto",
      torch_dtype=torch.float16,
      load_in_4bit=True,
      quantization_config=AWQConfig(...))
  ```

  のようになっている箇所は変更不要です（QwenモデルのロードにあたりFP16+4bitを指定）。注意点として、**SAMマスクデコーダのパラメータ**もできればfp16に合わせたほうが良いです。デコーダ部分はそれほど大きくないのでFP32でも動作しますが、統一のため`mask_decoder`や`prompt_encoder`のパラメータも`half()`に変換しておくと安全です。これは初期化直後に:

  ```python
  self.mask_decoder = self.mask_decoder.half()
  ```

  のようにするか、あるいは`config_qwen_sam.py`でTORCH\_DTYPEをfloat16に設定しているなら、それに従って型を合わせます。

* **学習・推論スクリプトの微調整:** Option Aに伴い、`train_minimal_lora.py`やテストスクリプトの**モデル生成部分**を修正します。従来は`create_sam_qwen_model(model_config)`内部でSAMエンコーダ読込等していましたが、それが無くなるため、**モデル生成が高速化**するかもしれません。基本的には`model = create_sam_qwen_model(config)`の呼び出し方や戻り値は変わらず、内部実装のみ変わる想定です。ただし、テストコードで**SAMエンコーダ関連の確認**（例えば `assert model.sam_image_encoder is not None` 等）をしている場合、それらは削除または`model.qwen_model.vision_tower`の存在確認に変更します。同様に、`config.USE_QFORMER`の意味合いが若干変化したため、テストでこのフラグを使って処理を分けているならロジックを点検します。

以上の変更を施した後、**ダミーデータセット**上で学習を実行し、以下を確認してください:

* 初期iterationでの**マスク損失および言語損失**に異常値がない（NaNや過大値が出ない）。
* 損失がエポックを通じて**減少傾向**を示す（Option Aへの変更によって学習が停滞しないこと）。
* `test_integrated_sam_qwen.py`を実行し、**推論結果**（生成テキストおよびマスク画像）が妥当かつ一貫して取得できる。

## 修正指示のまとめ (変更点リスト)

最後に、重要な修正点をファイル別に整理します。

* **model/sam\_qwen\_model.py**:

  * SAM画像エンコーダの初期化・利用部分を削除。代わりにQwenモデルのロードと参照を追加。
  * `forward`内で`image`入力を処理する際、`self.qwen_model.vision_tower`を用いて特徴取得し、MaskDecoderとLLMの双方に供給するよう変更。
  * MaskDecoderの出力を受け取り、アップサンプリング処理を追加。`results['mask']`の算出方法を更新。
  * Qwen LLMによるテキスト生成部分で、重複画像処理の回避策（モンキーパッチ or フック）を実装。`results['generated_text']`への格納を現行と同様に行う。

* **model/visual\_projector.py**:

  * `forward`のロジックをOption A用に変更。QwenのQ-Formerを使う場合は、このクラスは単なるラッパーとなるため、内部で`self.qwen_model`の該当部分を呼び出す実装にする。あるいはOptionAではVisualProjectorをバイパスする設計も可能なので、将来的にはクリーンアップ検討。

* **model/losses.py** (もし存在するなら):

  * マスク損失計算で、入力テンソルのdtypeやshapeに応じて微調整（必要なら）。例えば、MaskDecoderの出力をシグモイド通さず直接BCEWithLogitsLossに渡すなら、そのように。Option Aでマスク出力の値範囲が変わっていないか確認。

* **config\_qwen\_sam.py** および **train\_minimal\_lora.py**:

  * Option Aに合わせて設定値や処理を見直し。例えば`USE_QFORMER`はTrue固定でも良いかもしれない（Falseモードを使わないならコード簡素化）。
  * LoRA適用モジュールリストを、Qwenモデル名に沿ったものに更新（例: `"qwen_model.model.embed_tokens"` など正確な名前spaceはモデル構造に合わせる）。
  * （量子化関連設定は現行通り、変更不要）

* **その他**:

  * `test_integrated_sam_qwen.py`: Option AではSAMエンコーダが無くなるため、テスト内容でエンコーダ出力形状などを検証している場合アップデート。例えば「画像エンコーダ出力次元が期待通り」等のテストをしていたら、Qwenエンコーダの場合の値に変更。
  * ドキュメント (`README.md` 等): 統合フロー図を更新する。新フローは「1. 画像入力→**Qwen2.5-VL画像エンコーダ**→特徴、2. 特徴→**SAM2.1マスクデコーダ**+プロンプト→マスク、3. 特徴→**Qwen2.5-VL Q-Former**→LLM→テキスト生成」という形に書き換えます。またモンキーパッチの説明も「SAMエンコーダ→Qwen」に変えていた記述を削除し、「Qwenエンコーダを使用」に変更する。

以上の修正を順次適用し、**Option A**への移行を完了させてください。新アーキテクチャにより、将来のFoodLMM改に向けた画像と言語の深い統合理解と高精度な量推定が可能になることが期待されます。実装後は各種テストとデバッグを丁寧に行い、**現行ログ**との整合性（例えば初期loss挙動や推論結果の妥当性）が保たれていることを確認しましょう。
