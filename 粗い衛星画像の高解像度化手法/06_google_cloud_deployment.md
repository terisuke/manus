# 衛星画像超解像システム - Google Cloud実装ガイド

## 1. Google Cloud環境の概要

Google Cloud Platform (GCP) は、衛星画像超解像システムの実装に適した環境を提供します。このガイドでは、GCP上でのシステム実装方法、必要なサービスの設定、デプロイメント手順、および運用のベストプラクティスについて説明します。

## 2. 必要なGCPサービス

### 2.1 Google Compute Engine

高性能なGPUインスタンスを提供し、モデルの学習と推論に使用します。

```bash
# GPUインスタンスの作成
gcloud compute instances create satellite-sr-instance \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --restart-on-failure
```

### 2.2 Google Cloud Storage

データセット、学習済みモデル、および結果を保存するためのストレージサービスです。

```bash
# バケットの作成
gcloud storage buckets create gs://satellite-sr-data --location=us-central1

# データのアップロード
gcloud storage cp -r ./data gs://satellite-sr-data/
```

### 2.3 Google Kubernetes Engine (オプション)

大規模な処理のためのコンテナオーケストレーションを提供します。

```bash
# GKEクラスタの作成
gcloud container clusters create satellite-sr-cluster \
    --zone=us-central1-a \
    --num-nodes=3 \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --addons=HorizontalPodAutoscaling,HttpLoadBalancing
```

### 2.4 Vertex AI (オプション)

MLOpsとモデルデプロイメントを簡素化します。

## 3. 環境構築

### 3.1 開発環境のセットアップ

```bash
# SSHでインスタンスに接続
gcloud compute ssh satellite-sr-instance

# 必要なパッケージのインストール
sudo apt-get update
sudo apt-get install -y git python3-pip

# リポジトリのクローン
git clone https://github.com/your-repo/satellite-super-resolution.git
cd satellite-super-resolution

# 依存関係のインストール
pip install -r requirements.txt
```

### 3.2 GPUドライバとCUDAのセットアップ

PyTorch最新GPUイメージを使用する場合、GPUドライバとCUDAは事前にインストールされています。カスタムイメージを使用する場合は、以下の手順でセットアップします。

```bash
# GPUドライバのインストール
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# CUDAのパスを設定
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3.3 Dockerのセットアップ

```bash
# Dockerのインストール
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Dockerグループにユーザーを追加
sudo usermod -aG docker $USER
newgrp docker

# NVIDIA Docker Toolkitのインストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 4. データ管理

### 4.1 Google Cloud Storageとの連携

```python
from google.cloud import storage
import os

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """
    Google Cloud Storageからファイルをダウンロード
    
    Parameters:
    -----------
    bucket_name : str
        GCSバケット名
    source_blob_name : str
        GCS内のソースファイルパス
    destination_file_name : str
        ローカルの保存先パス
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)
    
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    ファイルをGoogle Cloud Storageにアップロード
    
    Parameters:
    -----------
    bucket_name : str
        GCSバケット名
    source_file_name : str
        ローカルのソースファイルパス
    destination_blob_name : str
        GCS内の保存先パス
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(source_file_name)
    
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")
```

### 4.2 データセットの準備

```python
def setup_dataset_on_gcp(bucket_name, dataset_prefix, local_data_dir):
    """
    GCP上でのデータセットのセットアップ
    
    Parameters:
    -----------
    bucket_name : str
        GCSバケット名
    dataset_prefix : str
        GCS内のデータセットのプレフィックス
    local_data_dir : str
        ローカルのデータディレクトリ
    """
    # GCSからデータをダウンロード
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # プレフィックスに一致するすべてのblobをリスト
    blobs = bucket.list_blobs(prefix=dataset_prefix)
    
    # ローカルディレクトリの作成
    os.makedirs(local_data_dir, exist_ok=True)
    
    # 各blobをダウンロード
    for blob in blobs:
        # プレフィックスを除いた相対パスを取得
        relative_path = blob.name[len(dataset_prefix):].lstrip('/')
        if not relative_path:  # ディレクトリ自体の場合はスキップ
            continue
            
        # ローカルの保存先パス
        destination_path = os.path.join(local_data_dir, relative_path)
        
        # 親ディレクトリの作成
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # ダウンロード
        blob.download_to_filename(destination_path)
    
    print(f"Dataset downloaded from gs://{bucket_name}/{dataset_prefix} to {local_data_dir}")
```

## 5. Dockerコンテナ化

### 5.1 Dockerfile

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Google Cloud SDKのインストール
RUN apt-get update && apt-get install -y curl apt-transport-https ca-certificates gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*

# アプリケーションコードのコピー
COPY . .

# コンテナ起動時のコマンド
CMD ["python", "train.py"]
```

### 5.2 requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.20.0
opencv-python>=4.5.0
scikit-image>=0.19.0
matplotlib>=3.5.0
tqdm>=4.60.0
pyyaml>=6.0
tensorboard>=2.10.0
google-cloud-storage>=2.0.0
```

### 5.3 Dockerイメージのビルドとプッシュ

```bash
# Dockerイメージのビルド
docker build -t gcr.io/your-project-id/satellite-sr:latest .

# Google Container Registryへのプッシュ
docker push gcr.io/your-project-id/satellite-sr:latest
```

## 6. 学習ジョブの実行

### 6.1 Google Compute Engineでの実行

```bash
# 学習スクリプトの実行
python train.py \
    --data_dir=/path/to/data \
    --batch_size=16 \
    --num_epochs=100 \
    --lr=1e-4 \
    --save_dir=/path/to/save \
    --gcs_bucket=satellite-sr-data \
    --gcs_output=models
```

### 6.2 Google Kubernetes Engineでの実行

```yaml
# kubernetes-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: satellite-sr-training
spec:
  template:
    spec:
      containers:
      - name: satellite-sr
        image: gcr.io/your-project-id/satellite-sr:latest
        args:
        - "python"
        - "train.py"
        - "--data_dir=/data"
        - "--batch_size=16"
        - "--num_epochs=100"
        - "--lr=1e-4"
        - "--save_dir=/output"
        - "--gcs_bucket=satellite-sr-data"
        - "--gcs_output=models"
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: output-volume
          mountPath: /output
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      - name: output-volume
        persistentVolumeClaim:
          claimName: output-pvc
      restartPolicy: Never
  backoffLimit: 4
```

```bash
# Kubernetesジョブの実行
kubectl apply -f kubernetes-job.yaml
```

### 6.3 Cloud Run Jobsでの実行

```bash
# Cloud Run Jobの作成
gcloud run jobs create satellite-sr-job \
    --image=gcr.io/your-project-id/satellite-sr:latest \
    --region=us-central1 \
    --memory=16Gi \
    --cpu=4 \
    --task-timeout=24h \
    --command="python" \
    --args="train.py,--gcs_bucket,satellite-sr-data,--gcs_dataset,datasets/satellite,--gcs_output,models,--num_epochs,100,--batch_size,16"

# ジョブの実行
gcloud run jobs execute satellite-sr-job --region=us-central1
```

## 7. モデルのデプロイ

### 7.1 Vertex AIでのモデルデプロイ

```python
from google.cloud import aiplatform

def deploy_model_to_vertex(model_path, model_name, machine_type="n1-standard-4", accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1):
    """
    モデルをVertex AIにデプロイ
    
    Parameters:
    -----------
    model_path : str
        GCS上のモデルパス
    model_name : str
        モデル名
    machine_type : str
        マシンタイプ
    accelerator_type : str
        アクセラレータタイプ
    accelerator_count : int
        アクセラレータ数
    
    Returns:
    --------
    str
        デプロイされたエンドポイントのID
    """
    # Vertex AIの初期化
    aiplatform.init(project='your-project-id', location='us-central1')
    
    # モデルのアップロード
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_path,
        serving_container_image_uri="gcr.io/your-project-id/satellite-sr-serving:latest"
    )
    
    # エンドポイントのデプロイ
    endpoint = model.deploy(
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        min_replica_count=1,
        max_replica_count=1
    )
    
    print(f"Model deployed to endpoint: {endpoint.name}")
    
    return endpoint.name
```

### 7.2 Cloud Runでのデプロイ

```bash
# サービングイメージのビルド
docker build -t gcr.io/your-project-id/satellite-sr-serving:latest -f Dockerfile.serving .

# イメージのプッシュ
docker push gcr.io/your-project-id/satellite-sr-serving:latest

# Cloud Runサービスのデプロイ
gcloud run deploy satellite-sr-service \
    --image=gcr.io/your-project-id/satellite-sr-serving:latest \
    --region=us-central1 \
    --memory=4Gi \
    --cpu=2 \
    --platform=managed \
    --allow-unauthenticated
```

### 7.3 サービング用Dockerfile

```dockerfile
# Dockerfile.serving
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.serving.txt .
RUN pip install --no-cache-dir -r requirements.serving.txt

# アプリケーションコードのコピー
COPY ./src /app/src
COPY ./models /app/models

# Flaskアプリケーションの設定
ENV PYTHONPATH=/app
ENV PORT=8080

# サービングアプリケーションの実行
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 src.serving.app:app
```

### 7.4 サービングアプリケーション

```python
# src/serving/app.py
import os
import io
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify
from PIL import Image
import base64

from src.models.swin2sr import Swin2SR
from src.inference.inference import preprocess_image, inference, postprocess_image

app = Flask(__name__)

# モデルの読み込み
model_path = os.environ.get('MODEL_PATH', '/app/models/best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Swin2SR(
    upscale=4,
    img_size=64,
    window_size=8,
    embed_dim=180,
    depths=[6, 6, 6, 6, 6, 6],
    num_heads=[6, 6, 6, 6, 6, 6],
    upsampler='pixelshuffle'
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    # 画像の読み込み
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 前処理
    lr_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 推論
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    # 後処理
    sr_img = sr_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    sr_img = np.clip(sr_img, 0, 1) * 255.0
    sr_img = sr_img.astype(np.uint8)
    
    # 画像をBase64エンコード
    _, buffer = cv2.imencode('.png', cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'image': img_str})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
```

## 8. 運用とモニタリング

### 8.1 Cloud Monitoringの設定

```python
from google.cloud import monitoring_v3

def setup_monitoring(project_id):
    """
    Cloud Monitoringの設定
    
    Parameters:
    -----------
    project_id : str
        GCPプロジェクトID
    """
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    # カスタムメトリクスの作成
    descriptor = monitoring_v3.MetricDescriptor()
    descriptor.type = "custom.googleapis.com/satellite_sr/inference_time"
    descriptor.metric_kind = monitoring_v3.MetricDescriptor.MetricKind.GAUGE
    descriptor.value_type = monitoring_v3.MetricDescriptor.ValueType.DOUBLE
    descriptor.description = "Time taken for super-resolution inference"
    
    descriptor = client.create_metric_descriptor(
        name=project_name,
        metric_descriptor=descriptor
    )
    
    print(f"Created {descriptor.name}")

def report_metric(project_id, value, instance_id):
    """
    メトリクスの報告
    
    Parameters:
    -----------
    project_id : str
        GCPプロジェクトID
    value : float
        メトリクス値
    instance_id : str
        インスタンスID
    """
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/satellite_sr/inference_time"
    series.resource.type = "gce_instance"
    series.resource.labels["instance_id"] = instance_id
    series.resource.labels["zone"] = "us-central1-a"
    
    point = series.points.add()
    point.value.double_value = value
    now = time.time()
    point.interval.end_time.seconds = int(now)
    
    client.create_time_series(
        name=project_name,
        time_series=[series]
    )
```

### 8.2 Cloud Loggingの設定

```python
import logging
from google.cloud import logging as cloud_logging

def setup_logging(project_id, log_name="satellite_sr"):
    """
    Cloud Loggingの設定
    
    Parameters:
    -----------
    project_id : str
        GCPプロジェクトID
    log_name : str
        ログ名
    """
    # Cloud Loggingクライアントの初期化
    client = cloud_logging.Client(project=project_id)
    
    # ロガーの設定
    logger = client.logger(log_name)
    
    # ハンドラーの設定
    handler = cloud_logging.handlers.CloudLoggingHandler(client, name=log_name)
    
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    
    return logger

def log_inference(logger, image_id, inference_time, success=True):
    """
    推論のログ記録
    
    Parameters:
    -----------
    logger : google.cloud.logging.Logger
        ロガー
    image_id : str
        画像ID
    inference_time : float
        推論時間
    success : bool
        成功したかどうか
    """
    logger.log_struct({
        "image_id": image_id,
        "inference_time": inference_time,
        "success": success
    })
```

## 9. コスト最適化

### 9.1 プリエンプティブルVMの使用

```bash
# プリエンプティブルVMの作成
gcloud compute instances create satellite-sr-preemptible \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --preemptible \
    --maintenance-policy=TERMINATE \
    --restart-on-failure
```

### 9.2 自動スケーリングの設定

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: satellite-sr-serving
spec:
  replicas: 1
  selector:
    matchLabels:
      app: satellite-sr-serving
  template:
    metadata:
      labels:
        app: satellite-sr-serving
    spec:
      containers:
      - name: satellite-sr
        image: gcr.io/your-project-id/satellite-sr-serving:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8080
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: satellite-sr-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: satellite-sr-serving
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 9.3 ストレージクラスの最適化

```bash
# 低コストのストレージクラスへの移行
gsutil rewrite -s NEARLINE gs://satellite-sr-data/archive/**
```

## 10. セキュリティ設定

### 10.1 IAMの設定

```bash
# サービスアカウントの作成
gcloud iam service-accounts create satellite-sr-sa \
    --display-name="Satellite SR Service Account"

# 権限の付与
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:satellite-sr-sa@your-project-id.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:satellite-sr-sa@your-project-id.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

### 10.2 VPCの設定

```bash
# VPCネットワークの作成
gcloud compute networks create satellite-sr-vpc --subnet-mode=auto

# ファイアウォールルールの設定
gcloud compute firewall-rules create satellite-sr-allow-internal \
    --network=satellite-sr-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/8
```

### 10.3 暗号化の設定

```bash
# カスタム暗号化キーの作成
gcloud kms keyrings create satellite-sr-keyring \
    --location=global

gcloud kms keys create satellite-sr-key \
    --location=global \
    --keyring=satellite-sr-keyring \
    --purpose=encryption

# バケットのCMEK設定
gsutil kms encryption -k projects/your-project-id/locations/global/keyRings/satellite-sr-keyring/cryptoKeys/satellite-sr-key gs://satellite-sr-data/
```

## 11. デプロイメントスクリプト

### 11.1 全体のデプロイメントスクリプト

```bash
#!/bin/bash
# deploy.sh

# 変数の設定
PROJECT_ID="your-project-id"
REGION="us-central1"
ZONE="${REGION}-a"
BUCKET_NAME="satellite-sr-data"
VM_NAME="satellite-sr-instance"
IMAGE_NAME="gcr.io/${PROJECT_ID}/satellite-sr:latest"
SERVING_IMAGE_NAME="gcr.io/${PROJECT_ID}/satellite-sr-serving:latest"

# プロジェクトの設定
gcloud config set project ${PROJECT_ID}

# バケットの作成
gcloud storage buckets create gs://${BUCKET_NAME} --location=${REGION}

# GPUインスタンスの作成
gcloud compute instances create ${VM_NAME} \
    --zone=${ZONE} \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --restart-on-failure

# Dockerイメージのビルドとプッシュ
docker build -t ${IMAGE_NAME} .
docker push ${IMAGE_NAME}

# サービングイメージのビルドとプッシュ
docker build -t ${SERVING_IMAGE_NAME} -f Dockerfile.serving .
docker push ${SERVING_IMAGE_NAME}

# Cloud Runサービスのデプロイ
gcloud run deploy satellite-sr-service \
    --image=${SERVING_IMAGE_NAME} \
    --region=${REGION} \
    --memory=4Gi \
    --cpu=2 \
    --platform=managed \
    --allow-unauthenticated

echo "Deployment completed successfully!"
```

### 11.2 学習ジョブの実行スクリプト

```bash
#!/bin/bash
# train.sh

# 変数の設定
PROJECT_ID="your-project-id"
REGION="us-central1"
BUCKET_NAME="satellite-sr-data"
IMAGE_NAME="gcr.io/${PROJECT_ID}/satellite-sr:latest"

# Cloud Run Jobの作成と実行
gcloud run jobs create satellite-sr-job \
    --image=${IMAGE_NAME} \
    --region=${REGION} \
    --memory=16Gi \
    --cpu=4 \
    --task-timeout=24h \
    --command="python" \
    --args="train.py,--gcs_bucket,${BUCKET_NAME},--gcs_dataset,datasets/satellite,--gcs_output,models,--num_epochs,100,--batch_size,16"

gcloud run jobs execute satellite-sr-job --region=${REGION}

echo "Training job submitted successfully!"
```

## 12. トラブルシューティング

### 12.1 一般的な問題と解決策

1. **GPUが認識されない**
   ```bash
   # NVIDIAドライバの確認
   nvidia-smi
   
   # CUDAのインストール確認
   nvcc --version
   
   # PyTorchのGPU確認
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **メモリ不足エラー**
   ```bash
   # スワップの設定
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

3. **Cloud Storageへのアクセスエラー**
   ```bash
   # 認証の確認
   gcloud auth list
   
   # サービスアカウントの設定
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

### 12.2 ログの確認

```bash
# Compute Engineのログ確認
gcloud compute ssh ${VM_NAME} --zone=${ZONE} -- "tail -f /var/log/syslog"

# Cloud Runのログ確認
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=satellite-sr-service" --limit=10
```

## 13. 参考リソース

- [Google Cloud Documentation](https://cloud.google.com/docs)
- [PyTorch on Google Cloud](https://cloud.google.com/pytorch/docs/pytorch-on-google-cloud)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Google Kubernetes Engine Documentation](https://cloud.google.com/kubernetes-engine/docs)
