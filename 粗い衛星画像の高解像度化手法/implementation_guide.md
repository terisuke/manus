# 衛星画像超解像システム - 実装手順書

## 概要

本ドキュメントは、NASAの衛星画像データを使用して解像度を5倍程度向上させる超解像システムの実装手順を説明します。Google Cloud環境での実装を前提としており、Swin2SRモデルとバイキュービック補間を組み合わせたアプローチを採用しています。

## 目次

1. [システム要件](#1-システム要件)
2. [環境構築](#2-環境構築)
3. [データ準備](#3-データ準備)
4. [モデル実装](#4-モデル実装)
5. [学習プロセス](#5-学習プロセス)
6. [推論と評価](#6-推論と評価)
7. [Google Cloud実装](#7-google-cloud実装)
8. [トラブルシューティング](#8-トラブルシューティング)

## 1. システム要件

### 1.1 機能要件

- NASAの衛星画像データを入力として受け取る
- 補完アルゴリズムを使用して解像度を5倍程度向上させる
- 衛星画像解析に適した高品質な超解像画像を生成する
- バッチ処理による複数画像の一括処理をサポート
- 推論結果の評価指標（PSNR、SSIM）を計算する

### 1.2 技術要件

- **プログラミング言語**: Python 3.8以上
- **深層学習フレームワーク**: PyTorch 2.0以上
- **主要ライブラリ**:
  - torchvision: 画像処理
  - numpy: 数値計算
  - opencv-python: 画像処理
  - scikit-image: 画像処理と評価指標
  - matplotlib: 可視化
  - tqdm: 進捗表示
  - google-cloud-storage: GCSとの連携
- **ハードウェア要件**:
  - 学習時: NVIDIA T4 GPU以上（推奨）
  - 推論時: CPU/GPU両対応

### 1.3 選定アルゴリズム

- **主要アプローチ**: Swin2SR (x1 → x4) + バイキュービック補間 (x4 → x5)
- **モデルアーキテクチャ**: Swin2SR（Swin Transformer V2ベースの超解像モデル）
- **損失関数**: SSIM Loss (1-SSIM)
- **データ拡張**: Brightness、Contrast、Flip、Rotate、CutMix
- **学習戦略**: 2段階学習（初期段階と微調整段階）

## 2. 環境構築

### 2.1 ローカル開発環境

```bash
# 仮想環境の作成
python -m venv satellite_sr_env
source satellite_sr_env/bin/activate  # Linux/Mac
# または
satellite_sr_env\Scripts\activate  # Windows

# 必要なパッケージのインストール
pip install torch torchvision torchaudio
pip install numpy opencv-python scikit-image matplotlib tqdm
pip install google-cloud-storage tensorboard
```

### 2.2 Google Cloud環境

```bash
# Google Cloud SDKのインストール
# https://cloud.google.com/sdk/docs/install

# プロジェクトの設定
gcloud config set project your-project-id

# Compute Engineインスタンスの作成
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

# SSHでインスタンスに接続
gcloud compute ssh satellite-sr-instance
```

### 2.3 プロジェクト構造

```
satellite-super-resolution/
├── data/
│   ├── train/
│   │   ├── lr/  # 低解像度画像
│   │   └── hr/  # 高解像度画像
│   ├── val/
│   │   ├── lr/
│   │   └── hr/
│   └── test/
│       ├── lr/
│       └── hr/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py  # データセットクラス
│   │   └── transforms.py  # データ変換
│   ├── models/
│   │   ├── __init__.py
│   │   └── swin2sr.py  # Swin2SRモデル
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py  # 損失関数
│   │   └── trainer.py  # 学習ループ
│   ├── inference/
│   │   ├── __init__.py
│   │   └── inference.py  # 推論処理
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py  # 評価指標
│   │   └── visualization.py  # 可視化
│   └── gcp/
│       ├── __init__.py
│       └── storage.py  # GCS連携
├── scripts/
│   ├── train.py  # 学習スクリプト
│   ├── inference.py  # 推論スクリプト
│   └── evaluate.py  # 評価スクリプト
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_visualization.ipynb
├── configs/
│   ├── train_config.yaml
│   └── inference_config.yaml
├── requirements.txt
├── Dockerfile
├── Dockerfile.serving
└── README.md
```

## 3. データ準備

### 3.1 NASAデータの取得

1. NASA Earthdata Search (https://search.earthdata.nasa.gov/) にアクセス
2. アカウント登録・ログイン
3. 以下の衛星データセットを検索:
   - MODIS (Moderate Resolution Imaging Spectroradiometer)
   - Landsat 8 OLI/TIRS
   - Sentinel-2 MSI
4. 必要な地域・期間のデータをダウンロード

### 3.2 データセットの作成

```python
# src/data/dataset.py
import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class SatelliteDataset(Dataset):
    """衛星画像データセット"""
    
    def __init__(self, lr_dir, hr_dir, transform=None, train=True):
        """
        Parameters:
        -----------
        lr_dir : str
            低解像度画像のディレクトリ
        hr_dir : str
            高解像度画像のディレクトリ
        transform : callable, optional
            データ拡張関数
        train : bool
            学習モードかどうか
        """
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        assert len(self.lr_files) == len(self.hr_files), "LRとHRの画像数が一致しません"
        
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        # 画像の読み込み
        lr_img = cv2.imread(self.lr_files[idx])
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        
        hr_img = cv2.imread(self.hr_files[idx])
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        # 正規化 (0-1の範囲に)
        lr_img = lr_img.astype(np.float32) / 255.0
        hr_img = hr_img.astype(np.float32) / 255.0
        
        # データ拡張
        if self.train and self.transform:
            # 同じ変換を適用するためにLRとHRを結合
            combined = np.concatenate([lr_img, hr_img], axis=2)
            combined = self.transform(combined)
            
            # 変換後に再分割
            lr_img = combined[:, :, :3]
            hr_img = combined[:, :, 3:]
        
        # NumPy配列からTensorに変換 (C, H, W形式に)
        lr_img = np.transpose(lr_img, (2, 0, 1))
        hr_img = np.transpose(hr_img, (2, 0, 1))
        
        return lr_img, hr_img
```

### 3.3 データ拡張

```python
# src/data/transforms.py
import numpy as np
import cv2

def transform_train(image):
    """
    学習用データ拡張
    
    Parameters:
    -----------
    image : np.ndarray
        入力画像
        
    Returns:
    --------
    np.ndarray
        拡張された画像
    """
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
    
    # 6. CutMix
    if np.random.random() > 0.5:
        # 画像の半分をランダムに切り取って入れ替える
        h, w = image.shape[:2]
        cut_ratio = np.random.uniform(0.2, 0.4)
        cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
        
        # ランダムな位置
        y = np.random.randint(0, h - cut_h)
        x = np.random.randint(0, w - cut_w)
        
        # 切り取り領域をランダムな値で埋める
        image[y:y+cut_h, x:x+cut_w] = np.random.uniform(0, 1, size=(cut_h, cut_w, image.shape[2]))
    
    return image
```

### 3.4 データローダー

```python
# src/data/dataloader.py
from torch.utils.data import DataLoader
from .dataset import SatelliteDataset
from .transforms import transform_train

def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    """
    データローダーを取得
    
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
    # 訓練用データセット
    train_dataset = SatelliteDataset(
        lr_dir=f"{data_dir}/train/lr",
        hr_dir=f"{data_dir}/train/hr",
        transform=transform_train,
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

## 4. モデル実装

### 4.1 Swin2SRモデル

```python
# src/models/swin2sr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class WindowAttention(nn.Module):
    """Windowベースのマルチヘッドセルフアテンション"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 相対位置バイアスのテーブル
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # 相対位置インデックスの取得
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # レイヤー
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # 相対位置バイアスの追加
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformerブロック"""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x, mask_matrix=None):
        B, L, C = x.shape
        H, W = int(np.sqrt(L)), int(np.sqrt(L))
        
        # 正規化
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # パディング
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # シフト
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        # ウィンドウ分割
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Windowアテンション
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # ウィンドウ結合
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # 逆シフト
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # パディング除去
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class Mlp(nn.Module):
    """MLP層"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    ウィンドウ分割
    
    Parameters:
    -----------
    x : torch.Tensor
        入力テンソル, shape: (B, H, W, C)
    window_size : int
        ウィンドウサイズ
        
    Returns:
    --------
    torch.Tensor
        ウィンドウ分割されたテンソル, shape: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    ウィンドウ結合
    
    Parameters:
    -----------
    windows : torch.Tensor
        ウィンドウ分割されたテンソル, shape: (num_windows*B, window_size, window_size, C)
    window_size : int
        ウィンドウサイズ
    H : int
        高さ
    W : int
        幅
        
    Returns:
    --------
    torch.Tensor
        結合されたテンソル, shape: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Swin2SR(nn.Module):
    """Swin2SR超解像モデル"""
    
    def __init__(self, upscale=4, img_size=64, window_size=8,
                 embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 mlp_ratio=2., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 upsampler='pixelshuffle', resi_connection='1conv'):
        super(Swin2SR, self).__init__()
        
        self.upscale = upscale
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.upsampler = upsampler
        
        # パッチ埋め込み
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=1, in_chans=3, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        # 位置埋め込み
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # ドロップパスレート
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Swinレイヤー
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformerLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=False)
            self.layers.append(layer)
        
        # 正規化
        self.norm = norm_layer(embed_dim)
        
        # 残差接続
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        
        # アップサンプラー
        if self.upsampler == 'pixelshuffle':
            self.upsample = UpsampleOneStep(upscale, embed_dim, 3)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x, x_size)
        
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, x_size[0], x_size[1])
        
        return x
    
    def forward(self, x):
        # 特徴抽出
        x = self.patch_embed.proj(x)
        x_shallow = x
        
        # Swin Transformer特徴
        x = self.forward_features(x)
        x = self.conv_after_body(x) + x_shallow
        
        # アップサンプリング
        x = self.upsample(x)
        
        return x

class SwinTransformerLayer(nn.Module):
    """Swin Transformerレイヤー"""
    
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2
        
        # ブロック
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        
        # ダウンサンプリング
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        
        # アテンションマスクの計算
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        # Transformerブロック
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        
        # ダウンサンプリング
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

class PatchEmbed(nn.Module):
    """パッチ埋め込み"""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # パディング
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))
        
        # 埋め込み
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        
        return x

class UpsampleOneStep(nn.Sequential):
    """1ステップアップサンプリング"""
    
    def __init__(self, scale, num_feat, num_out_ch, bias=True):
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1, bias=bias))
        m.append(nn.PixelShuffle(scale))
        
        super(UpsampleOneStep, self).__init__(*m)
```

### 4.2 バイキュービック補間

```python
# src/models/bicubic.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def bicubic_upsample(img, scale_factor):
    """
    バイキュービック補間によるアップサンプリング
    
    Parameters:
    -----------
    img : torch.Tensor or np.ndarray
        入力画像
    scale_factor : float
        スケールファクター
        
    Returns:
    --------
    torch.Tensor or np.ndarray
        アップサンプリングされた画像
    """
    if isinstance(img, torch.Tensor):
        # PyTorchテンソルの場合
        if img.dim() == 4:  # バッチ
            B, C, H, W = img.shape
            new_H, new_W = int(H * scale_factor), int(W * scale_factor)
            return F.interpolate(img, size=(new_H, new_W), mode='bicubic', align_corners=False)
        elif img.dim() == 3:  # 単一画像
            C, H, W = img.shape
            new_H, new_W = int(H * scale_factor), int(W * scale_factor)
            return F.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode='bicubic', align_corners=False).squeeze(0)
    else:
        # NumPy配列の場合
        if img.ndim == 4:  # バッチ
            B, H, W, C = img.shape
            new_H, new_W = int(H * scale_factor), int(W * scale_factor)
            result = np.zeros((B, new_H, new_W, C), dtype=img.dtype)
            for i in range(B):
                result[i] = cv2.resize(img[i], (new_W, new_H), interpolation=cv2.INTER_CUBIC)
            return result
        elif img.ndim == 3:  # 単一画像
            H, W, C = img.shape
            new_H, new_W = int(H * scale_factor), int(W * scale_factor)
            return cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
```

### 4.3 完全な超解像パイプライン

```python
# src/models/super_resolution.py
import torch
import torch.nn as nn
from .swin2sr import Swin2SR
from .bicubic import bicubic_upsample

class SuperResolutionPipeline(nn.Module):
    """
    完全な超解像パイプライン
    Swin2SR (x1 → x4) + バイキュービック補間 (x4 → x5)
    """
    
    def __init__(self, model_path=None, device='cuda'):
        super(SuperResolutionPipeline, self).__init__()
        
        # Swin2SRモデル
        self.swin2sr = Swin2SR(
            upscale=4,
            img_size=64,
            window_size=8,
            embed_dim=180,
            depths=[6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6],
            upsampler='pixelshuffle'
        )
        
        # 事前学習済みモデルの読み込み
        if model_path:
            self.swin2sr.load_state_dict(torch.load(model_path, map_location=device))
        
        self.swin2sr.to(device)
        self.device = device
    
    def forward(self, x):
        """
        順伝播
        
        Parameters:
        -----------
        x : torch.Tensor
            入力テンソル, shape: (B, C, H, W)
            
        Returns:
        --------
        torch.Tensor
            超解像テンソル, shape: (B, C, H*5, W*5)
        """
        # Swin2SRによる4倍超解像
        x_4x = self.swin2sr(x)
        
        # バイキュービック補間による1.25倍超解像（合計5倍）
        x_5x = bicubic_upsample(x_4x, scale_factor=1.25)
        
        return x_5x
```

## 5. 学習プロセス

### 5.1 損失関数

```python
# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SSIMLoss(nn.Module):
    """SSIM損失関数"""
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
        
    def _create_window(self, window_size, channel):
        """ガウシアンウィンドウの作成"""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _gaussian(self, window_size, sigma):
        """ガウシアンカーネルの作成"""
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def forward(self, img1, img2):
        """
        SSIM損失の計算
        
        Parameters:
        -----------
        img1 : torch.Tensor
            予測画像
        img2 : torch.Tensor
            ターゲット画像
            
        Returns:
        --------
        torch.Tensor
            1 - SSIM（損失値）
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
        
        return 1.0 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """SSIM値の計算"""
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class CombinedLoss(nn.Module):
    """SSIM損失とL1損失を組み合わせた複合損失関数"""
    
    def __init__(self, alpha=0.8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # SSIMの重み
        self.ssim_loss = SSIMLoss()
        
    def forward(self, x, y):
        ssim_loss = self.ssim_loss(x, y)
        l1_loss = F.l1_loss(x, y)
        
        # 重み付き結合
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss
```

### 5.2 トレーナー

```python
# src/training/trainer.py
import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=100, device='cuda', save_dir='./checkpoints', writer=None):
    """
    モデルの学習を行う関数
    
    Parameters:
    -----------
    model : nn.Module
        学習するモデル
    train_loader : DataLoader
        訓練用データローダー
    val_loader : DataLoader
        検証用データローダー
    criterion : nn.Module
        損失関数
    optimizer : torch.optim.Optimizer
        オプティマイザ
    scheduler : torch.optim.lr_scheduler._LRScheduler
        学習率スケジューラ
    num_epochs : int
        エポック数
    device : str
        使用するデバイス ('cuda' or 'cpu')
    save_dir : str
        モデルの保存ディレクトリ
    writer : SummaryWriter, optional
        TensorBoardのSummaryWriter
    
    Returns:
    --------
    tuple
        (学習済みモデル, 学習履歴)
    """
    # 保存ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    
    # 学習履歴
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_psnr': [],
        'val_ssim': []
    }
    
    # 最良のモデルを保存するための変数
    best_val_loss = float('inf')
    
    # デバイスの設定
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        
        for lr_imgs, hr_imgs in tqdm(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # 勾配のリセット
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(lr_imgs)
            
            # 損失の計算
            loss = criterion(outputs, hr_imgs)
            
            # 逆伝播と最適化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * lr_imgs.size(0)
        
        # エポックごとの平均損失
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # TensorBoardへの記録
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(val_loader):
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                # 順伝播
                outputs = model(lr_imgs)
                
                # 損失の計算
                loss = criterion(outputs, hr_imgs)
                val_loss += loss.item() * lr_imgs.size(0)
                
                # PSNR, SSIMの計算
                for i in range(outputs.size(0)):
                    output_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    hr_img = hr_imgs[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # 値の範囲を0-1に制限
                    output_img = np.clip(output_img, 0, 1)
                    
                    # PSNR
                    val_psnr_sum += psnr(hr_img, output_img, data_range=1.0)
                    
                    # SSIM
                    val_ssim_sum += ssim(hr_img, output_img, data_range=1.0, multichannel=True)
        
        # エポックごとの平均損失とメトリクス
        val_loss = val_loss / len(val_loader.dataset)
        val_psnr = val_psnr_sum / len(val_loader.dataset)
        val_ssim = val_ssim_sum / len(val_loader.dataset)
        
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        
        # TensorBoardへの記録
        if writer:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/PSNR', val_psnr, epoch)
            writer.add_scalar('Metrics/SSIM', val_ssim, epoch)
            
            # サンプル画像の記録
            if epoch % 10 == 0:
                # 最初のバッチから画像を取得
                for lr_imgs, hr_imgs in val_loader:
                    lr_imgs = lr_imgs.to(device)
                    hr_imgs = hr_imgs.to(device)
                    outputs = model(lr_imgs)
                    
                    # 最初の画像のみを使用
                    writer.add_images('Images/LR', lr_imgs[:4], epoch)
                    writer.add_images('Images/HR', hr_imgs[:4], epoch)
                    writer.add_images('Images/SR', outputs[:4], epoch)
                    break
        
        # 学習率の更新
        scheduler.step()
        
        # 結果の表示
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}')
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("Best model saved!")
        
        # 定期的なチェックポイントの保存
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth'))
    
    # 最終モデルの保存
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    return model, history
```

### 5.3 学習スクリプト

```python
# scripts/train.py
import os
import argparse
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.data.dataloader import get_dataloaders
from src.models.swin2sr import Swin2SR
from src.training.losses import SSIMLoss
from src.training.trainer import train_model
from src.utils.visualization import plot_learning_curves
from src.gcp.storage import upload_to_gcs

def main(args):
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データローダーの取得
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # モデルの初期化
    model = Swin2SR(
        upscale=4,
        img_size=args.patch_size,
        window_size=8,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        upsampler='pixelshuffle'
    )
    
    # 事前学習済みモデルの読み込み（存在する場合）
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained model from {args.pretrained_path}")
        model.load_state_dict(torch.load(args.pretrained_path), strict=False)
    
    # 損失関数の設定
    criterion = SSIMLoss()
    
    # オプティマイザの設定
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # スケジューラの設定
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)
    
    # 実験IDの生成
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, experiment_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # 設定の保存
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # TensorBoardの設定
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    
    # モデルの学習
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=save_dir,
        writer=writer
    )
    
    # 学習履歴の保存
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # 学習曲線のプロット
    plot_learning_curves(history, save_path=os.path.join(save_dir, 'learning_curves.png'))
    
    # GCSへのアップロード（指定されている場合）
    if args.gcs_bucket and args.gcs_output:
        print(f"Uploading results to GCS: gs://{args.gcs_bucket}/{args.gcs_output}")
        
        # モデルのアップロード
        upload_to_gcs(
            args.gcs_bucket,
            os.path.join(save_dir, 'best_model.pth'),
            f"{args.gcs_output}/{experiment_id}/best_model.pth"
        )
        
        # 設定のアップロード
        upload_to_gcs(
            args.gcs_bucket,
            os.path.join(save_dir, 'config.json'),
            f"{args.gcs_output}/{experiment_id}/config.json"
        )
        
        # 学習履歴のアップロード
        upload_to_gcs(
            args.gcs_bucket,
            os.path.join(save_dir, 'history.json'),
            f"{args.gcs_output}/{experiment_id}/history.json"
        )
        
        # 学習曲線のアップロード
        upload_to_gcs(
            args.gcs_bucket,
            os.path.join(save_dir, 'learning_curves.png'),
            f"{args.gcs_output}/{experiment_id}/learning_curves.png"
        )
    
    print(f"Training completed. Results saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin2SR model for satellite image super-resolution")
    
    # データ関連
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--patch_size', type=int, default=64, help='Training patch size')
    
    # モデル関連
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')
    
    # 学習関連
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # 保存関連
    parser.add_argument('--save_dir', type=str, default='./experiments', help='Directory to save results')
    
    # GCS関連
    parser.add_argument('--gcs_bucket', type=str, default=None, help='GCS bucket name')
    parser.add_argument('--gcs_output', type=str, default=None, help='GCS output directory')
    
    args = parser.parse_args()
    main(args)
```

## 6. 推論と評価

### 6.1 推論モジュール

```python
# src/inference/inference.py
import os
import cv2
import numpy as np
import torch
from src.models.super_resolution import SuperResolutionPipeline

def preprocess_image(image_path, device='cuda'):
    """
    推論のための画像前処理
    
    Parameters:
    -----------
    image_path : str
        入力画像のパス
    device : str
        使用するデバイス
        
    Returns:
    --------
    tuple
        (前処理された画像テンソル, 元の画像サイズ)
    """
    # 画像の読み込み
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 元のサイズを保存
    original_size = (img.shape[1], img.shape[0])  # (width, height)
    
    # 正規化 (0-1の範囲に)
    img = img.astype(np.float32) / 255.0
    
    # NumPy配列からTensorに変換
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    
    return img, original_size

def inference(model, lr_tensor, device='cuda'):
    """
    モデルを使用した推論
    
    Parameters:
    -----------
    model : nn.Module
        学習済みモデル
    lr_tensor : torch.Tensor
        低解像度画像テンソル
    device : str
        使用するデバイス
        
    Returns:
    --------
    torch.Tensor
        超解像画像テンソル
    """
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    return sr_tensor

def postprocess_image(sr_tensor, original_size=None):
    """
    推論結果の後処理
    
    Parameters:
    -----------
    sr_tensor : torch.Tensor
        超解像画像テンソル
    original_size : tuple, optional
        元の画像サイズ (width, height)
        
    Returns:
    --------
    np.ndarray
        後処理された画像 (RGB形式)
    """
    # Tensorから画像に変換
    sr_img = sr_tensor.squeeze(0).cpu().detach().numpy()
    sr_img = np.transpose(sr_img, (1, 2, 0))
    
    # 値の範囲を0-1に制限
    sr_img = np.clip(sr_img, 0, 1)
    
    # 元のサイズに戻す（オプション）
    if original_size is not None:
        # 5倍の解像度を考慮
        target_size = (original_size[0] * 5, original_size[1] * 5)
        sr_img = cv2.resize(sr_img, target_size, interpolation=cv2.INTER_CUBIC)
    
    # 0-255の範囲に変換
    sr_img = (sr_img * 255.0).astype(np.uint8)
    
    return sr_img

def super_resolve_image(model_path, image_path, output_path, device='cuda'):
    """
    画像の超解像処理を行う完全なパイプライン
    
    Parameters:
    -----------
    model_path : str
        モデルファイルのパス
    image_path : str
        入力画像のパス
    output_path : str
        出力画像の保存先パス
    device : str
        使用するデバイス ('cuda' or 'cpu')
        
    Returns:
    --------
    np.ndarray
        超解像画像
    """
    # モデルの読み込み
    model = SuperResolutionPipeline(model_path, device)
    model.eval()
    
    # 画像の前処理
    lr_tensor, original_size = preprocess_image(image_path, device)
    
    # 推論
    sr_tensor = inference(model, lr_tensor, device)
    
    # 後処理
    sr_img = postprocess_image(sr_tensor, original_size)
    
    # 結果の保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
    
    print(f"Super-resolved image saved to {output_path}")
    
    return sr_img
```

### 6.2 評価モジュール

```python
# src/utils/metrics.py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(sr_img, hr_img):
    """
    PSNR (Peak Signal-to-Noise Ratio) の計算
    
    Parameters:
    -----------
    sr_img : np.ndarray
        超解像画像
    hr_img : np.ndarray
        高解像度の参照画像
        
    Returns:
    --------
    float
        PSNR値 (dB)
    """
    # 画像のサイズが異なる場合はリサイズ
    if sr_img.shape != hr_img.shape:
        raise ValueError("Images must have the same dimensions")
    
    # 値の範囲を0-1に正規化
    sr_img_norm = sr_img.astype(np.float32) / 255.0
    hr_img_norm = hr_img.astype(np.float32) / 255.0
    
    # PSNR計算
    return peak_signal_noise_ratio(hr_img_norm, sr_img_norm, data_range=1.0)

def calculate_ssim(sr_img, hr_img):
    """
    SSIM (Structural Similarity Index) の計算
    
    Parameters:
    -----------
    sr_img : np.ndarray
        超解像画像
    hr_img : np.ndarray
        高解像度の参照画像
        
    Returns:
    --------
    float
        SSIM値 (0-1)
    """
    # 画像のサイズが異なる場合はリサイズ
    if sr_img.shape != hr_img.shape:
        raise ValueError("Images must have the same dimensions")
    
    # 値の範囲を0-1に正規化
    sr_img_norm = sr_img.astype(np.float32) / 255.0
    hr_img_norm = hr_img.astype(np.float32) / 255.0
    
    # SSIM計算
    return structural_similarity(hr_img_norm, sr_img_norm, data_range=1.0, multichannel=True)

def evaluate_super_resolution(sr_dir, hr_dir):
    """
    超解像結果の評価
    
    Parameters:
    -----------
    sr_dir : str
        超解像画像のディレクトリ
    hr_dir : str
        高解像度参照画像のディレクトリ
        
    Returns:
    --------
    dict
        評価結果
    """
    import os
    import cv2
    
    # 画像ファイルのリストを取得
    sr_files = sorted([f for f in os.listdir(sr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(sr_files) != len(hr_files):
        print(f"Warning: Number of SR images ({len(sr_files)}) does not match HR images ({len(hr_files)})")
    
    # 評価結果
    results = {
        'psnr': [],
        'ssim': [],
        'file_names': []
    }
    
    # 各画像ペアに対して評価
    for sr_file, hr_file in zip(sr_files, hr_files):
        sr_path = os.path.join(sr_dir, sr_file)
        hr_path = os.path.join(hr_dir, hr_file)
        
        # 画像の読み込み
        sr_img = cv2.imread(sr_path)
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
        
        hr_img = cv2.imread(hr_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        # 評価指標の計算
        psnr_value = calculate_psnr(sr_img, hr_img)
        ssim_value = calculate_ssim(sr_img, hr_img)
        
        # 結果の保存
        results['psnr'].append(psnr_value)
        results['ssim'].append(ssim_value)
        results['file_names'].append(sr_file)
        
        print(f"File: {sr_file}, PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
    
    # 平均値の計算
    results['avg_psnr'] = np.mean(results['psnr'])
    results['avg_ssim'] = np.mean(results['ssim'])
    
    print(f"Average PSNR: {results['avg_psnr']:.2f} dB")
    print(f"Average SSIM: {results['avg_ssim']:.4f}")
    
    return results
```

### 6.3 推論スクリプト

```python
# scripts/inference.py
import os
import argparse
import torch
import cv2
import numpy as np
from src.models.super_resolution import SuperResolutionPipeline
from src.inference.inference import preprocess_image, inference, postprocess_image
from src.utils.metrics import calculate_psnr, calculate_ssim
from src.utils.visualization import create_comparison_image, create_zoom_comparison
from src.gcp.storage import download_from_gcs, upload_to_gcs

def main(args):
    # デバイスの確認
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        device = 'cpu'
    
    # GCSからのモデルダウンロード（指定されている場合）
    if args.gcs_bucket and args.gcs_model:
        local_model_path = os.path.join('/tmp', os.path.basename(args.gcs_model))
        print(f"Downloading model from gs://{args.gcs_bucket}/{args.gcs_model}")
        download_from_gcs(args.gcs_bucket, args.gcs_model, local_model_path)
        model_path = local_model_path
    else:
        model_path = args.model
    
    # モデルの読み込み
    model = SuperResolutionPipeline(model_path, device)
    model.eval()
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # バッチ処理
    if args.batch:
        if not os.path.isdir(args.input):
            print("Error: Input must be a directory when using batch processing.")
            return
        
        os.makedirs(args.output, exist_ok=True)
        
        # 入力ディレクトリ内の画像ファイルを取得
        image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        for image_file in image_files:
            input_path = os.path.join(args.input, image_file)
            output_path = os.path.join(args.output, f"SR_{image_file}")
            
            # 画像の前処理
            lr_tensor, original_size = preprocess_image(input_path, device)
            
            # 推論
            sr_tensor = inference(model, lr_tensor, device)
            
            # 後処理
            sr_img = postprocess_image(sr_tensor, original_size)
            
            # 結果の保存
            cv2.imwrite(output_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
            
            print(f"Processed: {image_file} -> {output_path}")
        
        # 評価
        if args.evaluate and args.reference:
            if not os.path.isdir(args.reference):
                print("Error: Reference must be a directory when evaluating batch results.")
                return
            
            from src.utils.metrics import evaluate_super_resolution
            from src.utils.visualization import plot_evaluation_results
            
            results = evaluate_super_resolution(args.output, args.reference)
            
            # 評価結果のプロット
            plot_path = os.path.join(args.output, 'evaluation_results.png')
            plot_evaluation_results(results, save_path=plot_path)
    else:
        # 単一画像処理
        if not os.path.isfile(args.input):
            print("Error: Input must be a file when not using batch processing.")
            return
        
        # 画像の前処理
        lr_tensor, original_size = preprocess_image(args.input, device)
        
        # 推論
        sr_tensor = inference(model, lr_tensor, device)
        
        # 後処理
        sr_img = postprocess_image(sr_tensor, original_size)
        
        # 結果の保存
        cv2.imwrite(args.output, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
        
        print(f"Super-resolved image saved to {args.output}")
        
        # 比較画像の作成
        if args.compare:
            # 入力画像の読み込み
            lr_img = cv2.imread(args.input)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            
            if args.reference:
                # 参照画像の読み込み
                hr_img = cv2.imread(args.reference)
                hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                
                # 比較画像の作成
                comparison = create_comparison_image(lr_img, sr_img, hr_img)
                
                # 拡大比較画像の作成（オプション）
                if args.zoom:
                    zoom_comparison = create_zoom_comparison(lr_img, sr_img, hr_img)
                    
                    # 保存
                    zoom_path = os.path.join(os.path.dirname(args.output), 'zoom_comparison.png')
                    cv2.imwrite(zoom_path, cv2.cvtColor(zoom_comparison, cv2.COLOR_RGB2BGR))
                    print(f"Zoom comparison saved to {zoom_path}")
                
                # 評価
                if args.evaluate:
                    psnr_value = calculate_psnr(sr_img, hr_img)
                    ssim_value = calculate_ssim(sr_img, hr_img)
                    
                    print(f"PSNR: {psnr_value:.2f} dB")
                    print(f"SSIM: {ssim_value:.4f}")
            else:
                # 参照画像なしの比較
                comparison = create_comparison_image(lr_img, sr_img)
                
                # 拡大比較画像の作成（オプション）
                if args.zoom:
                    zoom_comparison = create_zoom_comparison(lr_img, sr_img)
                    
                    # 保存
                    zoom_path = os.path.join(os.path.dirname(args.output), 'zoom_comparison.png')
                    cv2.imwrite(zoom_path, cv2.cvtColor(zoom_comparison, cv2.COLOR_RGB2BGR))
                    print(f"Zoom comparison saved to {zoom_path}")
            
            # 比較画像の保存
            comparison_path = os.path.join(os.path.dirname(args.output), 'comparison.png')
            cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            print(f"Comparison image saved to {comparison_path}")
    
    # GCSへのアップロード（指定されている場合）
    if args.gcs_bucket and args.gcs_output:
        if args.batch:
            # バッチ処理の場合はディレクトリ全体をアップロード
            for root, _, files in os.walk(args.output):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, args.output)
                    gcs_path = f"{args.gcs_output}/{relative_path}"
                    upload_to_gcs(args.gcs_bucket, local_path, gcs_path)
        else:
            # 単一画像処理の場合は結果をアップロード
            upload_to_gcs(args.gcs_bucket, args.output, f"{args.gcs_output}/{os.path.basename(args.output)}")
            
            # 比較画像もアップロード（作成した場合）
            if args.compare:
                comparison_path = os.path.join(os.path.dirname(args.output), 'comparison.png')
                upload_to_gcs(args.gcs_bucket, comparison_path, f"{args.gcs_output}/comparison.png")
                
                if args.zoom:
                    zoom_path = os.path.join(os.path.dirname(args.output), 'zoom_comparison.png')
                    upload_to_gcs(args.gcs_bucket, zoom_path, f"{args.gcs_output}/zoom_comparison.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Super-resolution inference for satellite images')
    parser.add_argument('--model', type=str, help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output image or directory')
    parser.add_argument('--reference', type=str, default=None, help='Path to high-resolution reference image or directory')
    parser.add_argument('--batch', action='store_true', help='Process a batch of images')
    parser.add_argument('--compare', action='store_true', help='Create comparison images')
    parser.add_argument('--zoom', action='store_true', help='Create zoomed comparison images')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate results using reference images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    # GCS関連
    parser.add_argument('--gcs_bucket', type=str, default=None, help='GCS bucket name')
    parser.add_argument('--gcs_model', type=str, default=None, help='GCS model path')
    parser.add_argument('--gcs_output', type=str, default=None, help='GCS output directory')
    
    args = parser.parse_args()
    
    # モデルパスの確認
    if not args.model and not (args.gcs_bucket and args.gcs_model):
        parser.error("Either --model or both --gcs_bucket and --gcs_model must be specified")
    
    main(args)
```

## 7. Google Cloud実装

### 7.1 GCS連携モジュール

```python
# src/gcp/storage.py
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

def list_gcs_files(bucket_name, prefix=None):
    """
    Google Cloud Storage内のファイルをリスト
    
    Parameters:
    -----------
    bucket_name : str
        GCSバケット名
    prefix : str, optional
        ファイルプレフィックス
        
    Returns:
    --------
    list
        ファイルパスのリスト
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    
    return [blob.name for blob in blobs]
```

### 7.2 Dockerファイル

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
CMD ["python", "scripts/train.py"]
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

from src.models.super_resolution import SuperResolutionPipeline
from src.inference.inference import preprocess_image, inference, postprocess_image

app = Flask(__name__)

# モデルの読み込み
model_path = os.environ.get('MODEL_PATH', '/app/models/best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SuperResolutionPipeline(model_path, device)
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
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    lr_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
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

### 7.5 Cloud Run実装

```bash
#!/bin/bash
# deploy_cloud_run.sh

# 変数の設定
PROJECT_ID="your-project-id"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/satellite-sr-serving:latest"

# Dockerイメージのビルド
docker build -t ${IMAGE_NAME} -f Dockerfile.serving .

# Google Container Registryへのプッシュ
docker push ${IMAGE_NAME}

# Cloud Runサービスのデプロイ
gcloud run deploy satellite-sr-service \
    --image=${IMAGE_NAME} \
    --region=${REGION} \
    --memory=4Gi \
    --cpu=2 \
    --platform=managed \
    --allow-unauthenticated

echo "Cloud Run service deployed successfully!"
```

### 7.6 Vertex AI実装

```python
# scripts/deploy_vertex.py
import argparse
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

def main(args):
    endpoint_id = deploy_model_to_vertex(
        model_path=args.model_path,
        model_name=args.model_name,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count
    )
    
    print(f"Endpoint ID: {endpoint_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy model to Vertex AI')
    parser.add_argument('--model_path', type=str, required=True, help='GCS path to model directory')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--machine_type', type=str, default='n1-standard-4', help='Machine type')
    parser.add_argument('--accelerator_type', type=str, default='NVIDIA_TESLA_T4', help='Accelerator type')
    parser.add_argument('--accelerator_count', type=int, default=1, help='Number of accelerators')
    
    args = parser.parse_args()
    main(args)
```

## 8. トラブルシューティング

### 8.1 一般的な問題と解決策

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
   - バッチサイズを小さくする
   - 画像サイズを小さくする
   - 混合精度学習を使用する
   - 勾配チェックポイントを使用する

3. **学習が収束しない**
   - 学習率を調整する
   - 損失関数を変更する
   - データ拡張を見直す
   - モデルアーキテクチャを調整する

4. **推論が遅い**
   - モデルをTorchScriptに変換する
   - ONNXに変換する
   - TensorRTを使用する
   - バッチ処理を使用する

### 8.2 Google Cloud特有の問題

1. **認証エラー**
   ```bash
   # サービスアカウントの設定
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

2. **ディスク容量不足**
   ```bash
   # ディスクサイズの拡張
   gcloud compute disks resize DISK_NAME --size=SIZE_GB
   ```

3. **GPUクォータ不足**
   - Google Cloudコンソールからクォータ増加をリクエスト

4. **ネットワークエラー**
   - ファイアウォールルールを確認
   - VPCの設定を確認

### 8.3 デバッグ方法

1. **ログの確認**
   ```bash
   # Compute Engineのログ確認
   gcloud compute ssh VM_NAME -- "tail -f /var/log/syslog"
   
   # Cloud Runのログ確認
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=SERVICE_NAME"
   ```

2. **モデルの検証**
   ```python
   # 小さなテストデータでモデルをテスト
   with torch.no_grad():
       test_input = torch.randn(1, 3, 64, 64).to(device)
       test_output = model(test_input)
       print(f"Input shape: {test_input.shape}, Output shape: {test_output.shape}")
   ```

3. **メモリ使用量の確認**
   ```python
   # GPUメモリ使用量の確認
   import torch
   print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
   ```
