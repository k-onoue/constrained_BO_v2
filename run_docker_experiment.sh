#!/bin/bash

# ローカルの results, logs, temp, dbs ディレクトリを作成
mkdir -p results/
mkdir -p results/logs/
mkdir -p results/dbs/
mkdir -p temp/

# 各ディレクトリに必要な書き込み権限を付与
chmod -R 777 results/ temp/

# 現在の日付を取得（YYYY-MM-DD形式）
DATE="2024-10-25"

# Dockerfile の作成
dockerfile="Dockerfile"

dockerfile_content="
# 基本イメージとして公式の Python イメージを使用
FROM python:3.12

# 作業ディレクトリを設定
WORKDIR /app

# 必要なシステム依存ライブラリをインストール
RUN apt-get update && apt-get install -y \\
    pkg-config \\
    libgirepository1.0-dev \\
    gobject-introspection \\
    && rm -rf /var/lib/apt/lists/*

# 必要な Python パッケージをインストール (requirements.txt がある場合)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトファイルをすべてコピー
COPY . .

# run_experiments.sh の権限を付与
RUN chmod +x /app/run_experiments.sh

# コンテナ起動時に実行するコマンドを指定
CMD [\"bash\", \"/app/run_experiments.sh\"]
"

# Dockerfile を書き込み
echo "$dockerfile_content" > $dockerfile

# config.ini ファイルを Docker 内のパスに合わせて上書き
config_file="config.ini"

config_content="[paths]
project_dir = /app
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(results_dir)s/logs
dbs_dir = %(results_dir)s/dbs"

# config.ini ファイルを上書き
echo "$config_content" > $config_file

# run_experiments.sh を作成
run_script="run_experiments.sh"

run_script_content="#!/bin/bash
# 実験スクリプトを実行
bash /app/run_parafac_local.sh"

# run_experiments.sh に書き込み
echo "$run_script_content" > $run_script
chmod +x $run_script

# Dockerイメージのビルド
DOCKER_IMAGE="bo-env"
echo "Dockerイメージをビルドしています..."
docker build -t $DOCKER_IMAGE .

# Dockerでコンテナを実行し、results と logs ディレクトリをボリュームマウント
# ホストの results, logs, temp, config.ini をコンテナにマウント
echo "Dockerコンテナを実行しています..."
docker run --rm \
    -v "$(pwd)/results":/app/results \
    -v "$(pwd)/temp":/app/temp \
    -v "$(pwd)/config.ini":/app/config.ini \
    "$DOCKER_IMAGE"

# 実行完了のメッセージ
echo "Dockerコンテナが終了し、結果はローカルの results ディレクトリに保存されました。"

# Dockerイメージを削除
echo "Dockerイメージを削除しています..."
docker rmi "$DOCKER_IMAGE"