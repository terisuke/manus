# 衛星画像超解像システム - データパイプライン

## 1. データパイプラインの概要

データパイプラインは、NASAの衛星データを取得し、超解像モデルの学習と評価に適した形式に変換するプロセスを担当します。このモジュールは、データの取得、前処理、拡張、およびデータローダーの実装を含みます。

## 2. データ取得

### 2.1 NASAデータソースへのアクセス

```python
# NASA Earthdata APIを使用したデータ取得の例
import requests
import os
from urllib.parse import urlencode

def get_nasa_data(dataset, bbox, date_range, output_dir):
    """
    NASA Earthdataから衛星画像データを取得する関数
    
    Parameters:
    -----------
    dataset : str
        データセット識別子（例: 'MODIS_Terra_CorrectedReflectance_TrueColor'）
    bbox : tuple
        バウンディングボックス (min_lon, min_lat, max_lon, max_lat)
    date_range : tuple
        日付範囲 (start_date, end_date) - 'YYYY-MM-DD'形式
    output_dir : str
        出力ディレクトリ
        
    Returns:
    --------
    list
        ダウンロードされたファイルのパスのリスト
    """
    # NASA Earthdata APIのベースURL
    base_url = "https://search.earthdata.nasa.gov/search/granules"
    
    # APIパラメータの設定
    params = {
        'dataset_id': dataset,
        'bbox': ','.join(map(str, bbox)),
        'temporal': f"{date_range[0]},{date_range[1]}",
        'page_size': 100
    }
    
    # APIリクエスト
    response = requests.get(f"{base_url}?{urlencode(params)}")
    data = response.json()
    
    # ダウンロードリンクの抽出
    download_links = []
    for granule in data.get('feed', {}).get('entry', []):
        for link in granule.get('links', []):
            if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                download_links.append(link.get('href'))
    
    # データのダウンロード
    downloaded_files = []
    os.makedirs(output_dir, exist_ok=True)
    
    for i, link in enumerate(download_links):
        filename = os.path.join(output_dir, f"image_{i}.tif")
        response = requests.get(link, stream=True)
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        downloaded_files.append(filename)
        print(f"Downloaded: {filename}")
    
    return downloaded_files
```

### 2.2 Google Cloud Storageとの連携

```python
# Google Cloud Storageを使用したデータ管理
from google.cloud import storage

def upload_to_gcs(bucket_name, source_file, destination_blob):
    """
    ファイルをGoogle Cloud Storageにアップロードする関数
    
    Parameters:
    -----------
    bucket_name : str
        GCSバケット名
    source_file : str
        アップロードするローカルファイルのパス
    destination_blob : str
        GCS内の保存先パス
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    
    blob.upload_from_filename(source_file)
    
    print(f"File {source_file} uploaded to gs://{bucket_name}/{destination_blob}")

def download_from_gcs(bucket_name, source_blob, destination_file):
    """
    Google Cloud Storageからファイルをダウンロードする関数
    
    Parameters:
    -----------
    bucket_name : str
        GCSバケット名
    source_blob : str
        GCS内のソースファイルパス
    destination_file : str
        ダウンロード先のローカルファイルパス
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    
    os.makedirs(os.path.dirname(destination_file), exist_ok=True)
    blob.download_to_filename(destination_file)
    
    print(f"Blob {source_blob} downloaded to {destination_file}")
```

## 3. データ前処理

### 3.1 画像前処理

```python
# 画像前処理のための関数
import cv2
import numpy as np
from pathlib import Path

def preprocess_image(image_path, output_size=None):
    """
    衛星画像の前処理を行う関数
    
    Parameters:
    -----------
    image_path : str
        入力画像のパス
    output_size : tuple, optional
        リサイズする出力サイズ (width, height)
        
    Returns:
    --------
    np.ndarray
        前処理された画像
    """
    # 画像の読み込み
    img = cv2.imread(image_path)
    
    # BGR -> RGB変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 必要に応じてリサイズ
    if output_size is not None:
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_CUBIC)
    
    # 正規化 (0-1の範囲に)
    img = img.astype(np.float32) / 255.0
    
    return img

def create_lr_hr_pairs(hr_image, scale_factor=5):
    """
    高解像度画像から低解像度画像を生成し、ペアを作成する関数
    
    Parameters:
    -----------
    hr_image : np.ndarray
        高解像度画像
    scale_factor : int
        ダウンスケールする倍率
        
    Returns:
    --------
    tuple
        (低解像度画像, 高解像度画像)のペア
    """
    h, w, c = hr_image.shape
    lr_h, lr_w = h // scale_factor, w // scale_factor
    
    # 高解像度画像をダウンスケール
    lr_image = cv2.resize(hr_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    
    return lr_image, hr_image
```

### 3.2 データセット準備

```python
def prepare_dataset(hr_dir, output_dir, scale_factor=5, split_ratio=0.8):
    """
    データセットを準備する関数
    
    Parameters:
    -----------
    hr_dir : str
        高解像度画像のディレクトリ
    output_dir : str
        出力ディレクトリ
    scale_factor : int
        超解像の倍率
    split_ratio : float
        訓練/検証データの分割比率
    """
    hr_paths = list(Path(hr_dir).glob('*.jpg')) + list(Path(hr_dir).glob('*.png')) + list(Path(hr_dir).glob('*.tif'))
    
    # 出力ディレクトリの作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/train/lr").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/train/hr").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/val/lr").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/val/hr").mkdir(parents=True, exist_ok=True)
    
    # データをシャッフル
    np.random.shuffle(hr_paths)
    
    # 訓練/検証データの分割
    split_idx = int(len(hr_paths) * split_ratio)
    train_paths = hr_paths[:split_idx]
    val_paths = hr_paths[split_idx:]
    
    # 訓練データの処理
    for i, hr_path in enumerate(train_paths):
        hr_img = preprocess_image(str(hr_path))
        lr_img, hr_img = create_lr_hr_pairs(hr_img, scale_factor)
        
        # 保存
        cv2.imwrite(f"{output_dir}/train/lr/{i}.png", cv2.cvtColor(lr_img*255, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/train/hr/{i}.png", cv2.cvtColor(hr_img*255, cv2.COLOR_RGB2BGR))
    
    # 検証データの処理
    for i, hr_path in enumerate(val_paths):
        hr_img = preprocess_image(str(hr_path))
        lr_img, hr_img = create_lr_hr_pairs(hr_img, scale_factor)
        
        # 保存
        cv2.imwrite(f"{output_dir}/val/lr/{i}.png", cv2.cvtColor(lr_img*255, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/val/hr/{i}.png", cv2.cvtColor(hr_img*255, cv2.COLOR_RGB2BGR))
```

## 4. データ拡張

### 4.1 拡張手法

```python
def augment_data(image):
    """
    データ拡張を行う関数
    
    Parameters:
    -----------
    image : np.ndarray
        入力画像
        
    Returns:
    --------
    np.ndarray
        拡張された画像
    """
    # ランダムな変換を適用
    # 1. 水平反転
    if np.random.random() > 0.5:
        image = np.fliplr(image)
    
    # 2. 垂直反転
    if np.random.random() > 0.5:
        image = np.flipud(image)
    
    # 3. 90度単位の回転
    k = np.random.randint(0, 4)
    if k > 0:
        image = np.rot90(image, k)
    
    # 4. 明るさの調整
    if np.random.random() > 0.5:
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 1)
    
    # 5. コントラストの調整
    if np.random.random() > 0.5:
        contrast_factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * contrast_factor + mean, 0, 1)
    
    return image

def cutmix(img1, img2, alpha=0.5):
    """
    2つの画像にCutMixを適用
    
    Parameters:
    -----------
    img1, img2 : np.ndarray
        入力画像
    alpha : float
        ベータ分布のパラメータ
        
    Returns:
    --------
    np.ndarray
        CutMixが適用された画像
    """
    h, w, c = img1.shape
    
    # 混合比率をベータ分布からサンプリング
    lam = np.random.beta(alpha, alpha)
    
    # 切り取る領域のサイズを決定
    cut_h = int(h * np.sqrt(1. - lam))
    cut_w = int(w * np.sqrt(1. - lam))
    
    # 切り取る領域の開始位置をランダムに決定
    cy = np.random.randint(0, h - cut_h + 1)
    cx = np.random.randint(0, w - cut_w + 1)
    
    # 切り取った領域を入れ替え
    mixed_img = img1.copy()
    mixed_img[cy:cy+cut_h, cx:cx+cut_w, :] = img2[cy:cy+cut_h, cx:cx+cut_w, :]
    
    return mixed_img
```

## 5. データローダー

### 5.1 PyTorchデータセット

```python
# PyTorchデータセットの実装
import torch
from torch.utils.data import Dataset, DataLoader
import os

class SatelliteDataset(Dataset):
    """衛星画像のデータセット"""
    
    def __init__(self, lr_dir, hr_dir, transform=None, train=True):
        """
        Parameters:
        -----------
        lr_dir : str
            低解像度画像のディレクトリ
        hr_dir : str
            高解像度画像のディレクトリ
        transform : callable, optional
            データ拡張のための変換関数
        train : bool
            訓練モードかどうか
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.train = train
        
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))
        
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        
        # 画像の読み込み
        lr_img = cv2.imread(lr_path)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_img = lr_img.astype(np.float32) / 255.0
        
        hr_img = cv2.imread(hr_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        hr_img = hr_img.astype(np.float32) / 255.0
        
        # 訓練時のデータ拡張
        if self.train and self.transform:
            # 同じ変換を両方の画像に適用
            seed = np.random.randint(2147483647)
            
            np.random.seed(seed)
            lr_img = self.transform(lr_img)
            
            np.random.seed(seed)
            hr_img = self.transform(hr_img)
        
        # NumPy配列からTensorに変換
        lr_tensor = torch.from_numpy(np.transpose(lr_img, (2, 0, 1)))
        hr_tensor = torch.from_numpy(np.transpose(hr_img, (2, 0, 1)))
        
        return lr_tensor, hr_tensor
```

### 5.2 データローダー

```python
def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    """
    訓練用と検証用のデータローダーを取得する関数
    
    Parameters:
    -----------
    data_dir : str
        データセットのルートディレクトリ
    batch_size : int
        バッチサイズ
    num_workers : int
        データローダーのワーカー数
        
    Returns:
    --------
    tuple
        (訓練用データローダー, 検証用データローダー)
    """
    # データ拡張関数
    def transform(image):
        return augment_data(image)
    
    # 訓練用データセット
    train_dataset = SatelliteDataset(
        lr_dir=f"{data_dir}/train/lr",
        hr_dir=f"{data_dir}/train/hr",
        transform=transform,
        train=True
    )
    
    # 検証用データセット
    val_dataset = SatelliteDataset(
        lr_dir=f"{data_dir}/val/lr",
        hr_dir=f"{data_dir}/val/hr",
        transform=None,
        train=False
    )
    
    # データローダー
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

## 6. データパイプラインの使用例

```python
# データパイプラインの使用例
def main():
    # 1. NASAのデータを取得
    nasa_data = get_nasa_data(
        dataset='MODIS_Terra_CorrectedReflectance_TrueColor',
        bbox=(-122.5, 37.5, -122.0, 38.0),  # サンフランシスコ周辺
        date_range=('2023-01-01', '2023-01-31'),
        output_dir='./raw_data'
    )
    
    # 2. データセットの準備
    prepare_dataset(
        hr_dir='./raw_data',
        output_dir='./dataset',
        scale_factor=5,
        split_ratio=0.8
    )
    
    # 3. データローダーの取得
    train_loader, val_loader = get_dataloaders(
        data_dir='./dataset',
        batch_size=16,
        num_workers=4
    )
    
    # 4. データの確認
    for lr_imgs, hr_imgs in train_loader:
        print(f"LR shape: {lr_imgs.shape}, HR shape: {hr_imgs.shape}")
        break

if __name__ == "__main__":
    main()
```

## 7. Google Cloud上でのデータパイプライン

```python
# Google Cloud上でのデータパイプラインの実装例
def setup_gcp_data_pipeline(gcs_bucket, dataset_prefix, local_data_dir):
    """
    Google Cloud上でのデータパイプラインのセットアップ
    
    Parameters:
    -----------
    gcs_bucket : str
        GCSバケット名
    dataset_prefix : str
        GCS内のデータセットのプレフィックス
    local_data_dir : str
        ローカルのデータディレクトリ
    
    Returns:
    --------
    tuple
        (訓練用データローダー, 検証用データローダー)
    """
    # GCSからデータをダウンロード
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket)
    
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
    
    # データセットの準備
    prepare_dataset(
        hr_dir=local_data_dir,
        output_dir=os.path.join(local_data_dir, 'processed'),
        scale_factor=5,
        split_ratio=0.8
    )
    
    # データローダーの取得
    train_loader, val_loader = get_dataloaders(
        data_dir=os.path.join(local_data_dir, 'processed'),
        batch_size=16,
        num_workers=4
    )
    
    return train_loader, val_loader
```
