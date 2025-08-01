# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
日本語で応答すること！
Webリサーチはo3をタイムアウトの設定なしで用いること！

[Introduction]
将来的な目標：LISA→FoodLMMの成功を受けて、最新のVLMと最新のSAMを統合して、LISAのような深い次元で画像と言語を理解した基盤モデル（LISA改）を作り、それをFoodLMMの学習方法を参考にファインチューニングし、FoodLMM改を作り、写真内の料理や食材の量の推定を精度高く行わせる予定。VLMとSAMのようなモデルの統合モデルについて、最新の知見も導入しながらポテンシャルの高いモデル実装を目指している。

現状：LISA改を作っている。Llama4-LISA-Code/にはLlama4 scoutとSAM2のQformerを介した統合モデルを実装し、test(Llama4-LISA-Code/test_phase3b_enhanced_multiscale_lora.py)が問題なく走る（ログ：Llama4-LISA-Code/logs/202507271910.log）ところまで完了した。

方針転換：Qwen2.5-VL-72BとSAM2.1の統合モデルをLISA改とする方針に転換したい。まずはQwen2.5-VL-3BとSAM2.1の統合モデル（SAM2_1_Qwen_2_5）として進め、上記のようなtest scriptで問題がなく走るか確かめたい。


現状、model, utilsフォルダにはLlama4-LISA-Codeのmodel, utilsフォルダと同様のものを配置している。また、test_phase3b_enhanced_multiscale_lora.pyも同様のものを配置している。これから、既存のmodel, utilsフォルダやtest_phase3b_enhanced_multiscale_lora.pyで呼び出しているファイルを参考にしつつ、またmd_files/o3_spec.mdに記載の部分はmd_files/o3_spec.mdを参考にしつつ、SAM2_1_Qwen_2_5の仕様に変更してほしい（ファイルの作成や削除を行なってほしい）。test_phase3b_enhanced_multiscale_lora.pyですら使わないファイルがmodelフォルダにはあるはずなので、不必要なファイルは削除してSAM2_1_Qwen_2_5の仕様に合わせてシンプルにしてほしい。

→md_files/past_files/o3_spec3mdの方針で実装を進め、現状md_files/implementation_archiveのような状況でtest_integrated_sam_qwen.pyが一通り走った。

[命令]
上記の方針で実装を進めてきた。現状のコードを理解して、md_files/current_spec/o3_spec4.mdを参照して追加実装・修正してほしい。

※実装の際に注意すること
・自信のない部分は適宜正規（Qwen, SAM, Qformer, Huggingface, Pytorchなど公式の実装）の実装をWebでしらべながら予想や自前の実装を少なくして実装すること
・o3_spec4.mdはあくまでおおまかな指針であるので、細かな実装はWebでベストプラクティスをリサーチして[Introduction]に述べている目的に沿うように、本質的に実装を進めること（簡易な実装でとりあえず走るコードは必要ない、本質的に目標を達成するコードが欲しい）
・Webリサーチを積極的に行い、Qwen, SAMの正規の実装をできるだけ用いること
・フォールバック的もしくはダミーコードはエラーを隠蔽するので適切にエラーを出して止め、次のデバッグに繋がる情報を提供するように修正すること
・解決した問題に関するデバッグログや必要のないデバッグログは削除、デバッグ作業により解決した部分はシンプル（分岐のない確定的な実装）にして、コードをシンプルに保つようにすること




## Development Rules & Guidelines

### ⭐️デバッグ修正ルール：以下のフローに沿って実装すること
1. エラーの原因と本質的な修正方針をざっくり推定
2. 仕様書（md_files/o3_spec.md）に沿った実装はどのようなものかを考える。
3. その上で、Webリサーチが必要であれば行う（必要なければスキップ）
4. デバッグコードを用いたデバッグが必要であれば行う（必要なければスキップ）
5. すでに走ったtest script（今回は、なし）の内容が参考になりそうか考える
6. 1-5を踏まえて、再度本質的な修正方針を考え、一度提案する
7. 提案に対して私が許可もしくはどの提案を採用するか判断するので、その判断に基づいて修正する
※開発や修正において、対症療法的な処理やフォールバック的な機能はエラーを隠蔽するので、本質的な解決につなげるため、エラーを出して止めて次のデバッグに繋げるように開発すること