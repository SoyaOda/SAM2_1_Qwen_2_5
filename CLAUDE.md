# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
日本語で応答すること！
o3によるリサーチは行わずwebでシンプルにリサーチすること！

[Introduction]
md_filesを元に実装してきた（まだ未実装もあり）。

[命令]
まず、train_phase3b_enhanced.pyで呼び出しているモデルや関連ファイルをよく読んで内容を理解して。

md_files/todo/training_roadmap_20250127.mdを参考に、train_phase3b_enhanced.pyを実装中。

※適宜webでllama4やSAM2の仕様、サンプルコード、ベストプラクティスを調べつつtodoや修正方針を作り、修正して 

※フォールバック的なコードやダミーコードはエラーを隠蔽するので、シンプルにエラーを出して止めるように実装すること




## Development Rules & Guidelines

### ⭐️開発方針作成ルール：以下のフローに沿って実装すること
1. 与えられたお題に対して、Webリサーチはを行う
2. 1を元にcurrent_dev_spec_md_fileに実装の仕様書をmdファイルとして保存する。
3. todoリストも同様に作成する
4. 2, 3のmdファイルとtodoリストを元に開発を進める

### ⭐️デバッグ修正ルール：以下のフローに沿って実装すること
1. エラーの原因と本質的な修正方針をざっくり推定
2. 仕様書（md_files/current/o3-modification20250727.md）に沿った実装はどのようなものかを考える。
3. その上で、Webリサーチはが必要そうか考える
4. デバッグコードを用いたデバッグが必要そうか考える
5. すでに走ったtest script（今回は、なし）の内容が参考になりそうか考える
6. 1-5を踏まえて、再度本質的な修正方針を考え、一度提案する
7. 提案に対して私が許可もしくはどの提案を採用するか判断するので、その判断に基づいて修正する
※開発や修正において、フォールバック的な機能はエラーを隠蔽するので、エラーを出して止めて次のデバッグに繋げるように開発すること


### Lambda Cloud Development Workflow
Lambda Cloud環境での実行を行うので、ローカルファイルの修正を行うたびに、以下のコマンド例を参考に、lambda上に転送し、lambda上で実行すること

※タイムアウトは設定しないこと！

```bash
# File transfer to Lambda Cloud
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@<ip address>:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# Remote execution
ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip address> "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1, ... PYTHONUNBUFFERED=1 python -u <script>.py 2>&1"

or 

ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip address> "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 python -u <script>.py 2>&1"
```
ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.53.149 "cd /lambda/nfs/llama4-lisa-project-fs-north-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export CUDA_LAUNCH_BLOCKING=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && PYTHONUNBUFFERED=1 python -u test_phase3b_integration_real.py 2>&1"