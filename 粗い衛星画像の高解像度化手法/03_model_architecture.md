# 衛星画像超解像システム - モデルアーキテクチャ

## 1. モデルアーキテクチャの概要

本システムでは、Swin2SRモデルをベースとした超解像アーキテクチャを採用します。Swin2SRは、Swinトランスフォーマーをベースにした最新の超解像モデルで、従来のCNNベースのモデルよりも優れた性能を示しています。また、4倍の超解像後にバイキュービック補間を適用して、合計5倍の解像度向上を実現します。

## 2. Swin2SRモデル

### 2.1 モデル構造

Swin2SRモデルは以下の主要コンポーネントで構成されています：

1. **浅い特徴抽出**：入力画像から初期特徴を抽出
2. **深い特徴抽出**：Swinトランスフォーマーブロックによる特徴処理
3. **アップサンプリング**：PixelShuffleによる解像度の向上
4. **最終出力層**：超解像画像の生成

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Swin2SR(nn.Module):
    """Swin2SRモデル"""
    
    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=180, 
                 depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 upscale=4, upsampler='pixelshuffle', resi_connection='1conv'):
        super().__init__()
        self.upscale = upscale
        self.upsampler = upsampler
        
        # 浅い特徴抽出
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # 深い特徴抽出（Swinトランスフォーマーブロック）
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patches_resolution = self.patch_embed.patches_resolution
        
        # Swinトランスフォーマーブロックの実装
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
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                downsample=None
            )
            self.layers.append(layer)
        
        # 正規化層
        self.norm = norm_layer(embed_dim)
        
        # アップサンプリング
        if self.upsampler == 'pixelshuffle':
            # 4倍アップサンプリング
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 4 * embed_dim, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(embed_dim, 4 * embed_dim, 3, 1, 1),
                nn.PixelShuffle(2)
            )
        else:
            # その他のアップサンプリング方法
            self.upsample = nn.Identity()
        
        # 最終出力層
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        
        # 初期化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        
        # Swinトランスフォーマーブロックを通す
        for layer in self.layers:
            x = layer(x, x_size)
        
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(-1, self.embed_dim, x_size[0], x_size[1])
        
        return x
    
    def forward(self, x):
        # 入力サイズの取得
        H, W = x.shape[2:]
        
        # 浅い特徴抽出
        x_first = self.conv_first(x)
        
        # 深い特徴抽出（Swinトランスフォーマーブロック）
        x_deep = self.forward_features(x_first)
        
        # 残差接続
        x_deep = x_deep + x_first
        
        # アップサンプリング
        x_up = self.upsample(x_deep)
        
        # 最終出力
        x_out = self.conv_last(x_up)
        
        # 5倍にするための追加のバイキュービック補間
        if self.upscale == 4:  # 4倍から5倍へ
            x_out = F.interpolate(x_out, scale_factor=1.25, mode='bicubic', align_corners=False)
        
        return x_out
```

### 2.2 主要コンポーネント

#### 2.2.1 パッチ埋め込み

```python
class PatchEmbed(nn.Module):
    """画像をパッチに分割し、埋め込みベクトルに変換するモジュール"""
    
    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=180):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x
```

#### 2.2.2 ウィンドウアテンション

```python
class WindowAttention(nn.Module):
    """ウィンドウベースのマルチヘッドセルフアテンション"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 相対位置バイアスのためのパラメータテーブル
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # 相対位置インデックスの計算
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # シフトして正の値にする
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        # QKV変換
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 初期化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 入力特徴量 (B*nW, N, C)
            mask: アテンションマスク (nW, N, N)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C/nH
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N
        
        # 相対位置バイアスの追加
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # マスクがある場合は適用
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

#### 2.2.3 Swinトランスフォーマーブロック

```python
class SwinTransformerBlock(nn.Module):
    """Swinトランスフォーマーブロック"""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x, mask_matrix=None):
        B, L, C = x.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        
        # 残差接続
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循環シフト
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        # ウィンドウ分割
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        # Windowアテンション
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        
        # ウィンドウ結合
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        
        # 逆循環シフト
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
```

#### 2.2.4 Swinトランスフォーマー層

```python
class SwinTransformerLayer(nn.Module):
    """Swinトランスフォーマー層"""
    
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Swinトランスフォーマーブロックの構築
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # ダウンサンプリング層（オプション）
        self.downsample = downsample
        
    def forward(self, x, x_size):
        # アテンションマスクの計算
        H, W = x_size
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        
        # パディングが必要な場合
        if H != Hp or W != Wp:
            # パディングの実装（簡略化のため省略）
            pass
        
        # 各ブロックを通す
        for blk in self.blocks:
            x = blk(x, None)  # アテンションマスクは簡略化のためNoneとする
        
        # ダウンサンプリング（存在する場合）
        if self.downsample is not None:
            x = self.downsample(x, x_size)
        
        return x
```

#### 2.2.5 ヘルパー関数

```python
# ウィンドウ分割と結合のヘルパー関数
def window_partition(x, window_size):
    """入力テンソルをウィンドウに分割"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """ウィンドウを元のテンソル形状に戻す"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
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

class DropPath(nn.Module):
    """ドロップパス: 確率pでパスをドロップする"""
    
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二値化
        output = x.div(keep_prob) * random_tensor
        return output
```

## 3. バイキュービック補間

4倍から5倍への追加の拡大には、バイキュービック補間を使用します。

```python
def bicubic_upsample(image, scale_factor=1.25):
    """
    バイキュービック補間による画像の拡大
    
    Parameters:
    -----------
    image : torch.Tensor
        入力画像 (B, C, H, W)
    scale_factor : float
        拡大倍率
        
    Returns:
    --------
    torch.Tensor
        拡大された画像 (B, C, H*scale_factor, W*scale_factor)
    """
    return F.interpolate(image, scale_factor=scale_factor, mode='bicubic', align_corners=False)
```

## 4. モデルの初期化と読み込み

```python
def initialize_model(upscale=4, pretrained_path=None, device='cuda'):
    """
    モデルの初期化と事前学習済みの重みの読み込み
    
    Parameters:
    -----------
    upscale : int
        アップスケール倍率（デフォルト: 4）
    pretrained_path : str, optional
        事前学習済みモデルのパス
    device : str
        使用するデバイス ('cuda' or 'cpu')
        
    Returns:
    --------
    nn.Module
        初期化されたモデル
    """
    # モデルの初期化
    model = Swin2SR(
        upscale=upscale,
        img_size=64,
        window_size=8,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        upsampler='pixelshuffle'
    )
    
    # 事前学習済みモデルの読み込み（存在する場合）
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    # デバイスに転送
    model = model.to(device)
    
    return model
```

## 5. モデルの使用例

```python
def super_resolve_image(model, lr_image, device='cuda'):
    """
    低解像度画像から超解像画像を生成
    
    Parameters:
    -----------
    model : nn.Module
        学習済みSwin2SRモデル
    lr_image : np.ndarray
        低解像度入力画像 (RGB形式)
    device : str
        使用するデバイス ('cuda' or 'cpu')
        
    Returns:
    --------
    np.ndarray
        超解像画像 (RGB形式)
    """
    # 前処理
    lr_tensor = preprocess_for_inference(lr_image, device)
    
    # 推論
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    # 後処理
    sr_image = postprocess_from_inference(sr_tensor)
    
    return sr_image

def preprocess_for_inference(image, device):
    """
    推論のための画像前処理
    
    Parameters:
    -----------
    image : np.ndarray
        入力画像 (RGB形式)
    device : str
        使用するデバイス
        
    Returns:
    --------
    torch.Tensor
        前処理された画像テンソル
    """
    # NumPy配列からTensorに変換
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    
    return image

def postprocess_from_inference(tensor):
    """
    推論結果の後処理
    
    Parameters:
    -----------
    tensor : torch.Tensor
        モデル出力テンソル
        
    Returns:
    --------
    np.ndarray
        後処理された画像 (RGB形式)
    """
    # Tensorから画像に変換
    image = tensor.squeeze(0).cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1) * 255.0
    image = image.astype(np.uint8)
    
    return image
```

## 6. モデルの評価

```python
def evaluate_model(model, test_loader, device='cuda'):
    """
    モデルの評価
    
    Parameters:
    -----------
    model : nn.Module
        評価するモデル
    test_loader : DataLoader
        テスト用データローダー
    device : str
        使用するデバイス ('cuda' or 'cpu')
        
    Returns:
    --------
    dict
        評価結果 (PSNR, SSIM)
    """
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # 推論
            sr_imgs = model(lr_imgs)
            
            # 評価指標の計算
            for i in range(sr_imgs.size(0)):
                sr_img = sr_imgs[i].cpu().numpy().transpose(1, 2, 0)
                hr_img = hr_imgs[i].cpu().numpy().transpose(1, 2, 0)
                
                # 値の範囲を0-1に制限
                sr_img = np.clip(sr_img, 0, 1)
                
                # PSNR
                psnr_value = peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)
                psnr_values.append(psnr_value)
                
                # SSIM
                ssim_value = structural_similarity(hr_img, sr_img, data_range=1.0, multichannel=True)
                ssim_values.append(ssim_value)
    
    # 平均値の計算
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }
```

## 7. モデルの保存と読み込み

```python
def save_model(model, save_path):
    """
    モデルの保存
    
    Parameters:
    -----------
    model : nn.Module
        保存するモデル
    save_path : str
        保存先のパス
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path, device='cuda'):
    """
    モデルの読み込み
    
    Parameters:
    -----------
    model : nn.Module
        読み込み先のモデル
    load_path : str
        読み込むモデルのパス
    device : str
        使用するデバイス ('cuda' or 'cpu')
        
    Returns:
    --------
    nn.Module
        読み込まれたモデル
    """
    model.load_state_dict(torch.load(load_path, map_location=device))
    model = model.to(device)
    return model
```

## 8. モデルの最適化

### 8.1 混合精度学習

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(model, train_loader, criterion, optimizer, device):
    """混合精度学習を使用したトレーニングループ"""
    model.train()
    scaler = GradScaler()
    
    for lr_imgs, hr_imgs in train_loader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度での順伝播
        with autocast():
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
        
        # スケーリングされた勾配の計算と更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 8.2 勾配チェックポイント

```python
from torch.utils.checkpoint import checkpoint

class SwinTransformerBlockWithCheckpoint(SwinTransformerBlock):
    """勾配チェックポイントを使用したSwinトランスフォーマーブロック"""
    
    def forward(self, x, mask_matrix=None):
        def forward_fn(x_input, mask_input):
            return super(SwinTransformerBlockWithCheckpoint, self).forward(x_input, mask_input)
        
        return checkpoint(forward_fn, x, mask_matrix)
```

## 9. モデルのデプロイ

```python
def export_onnx_model(model, save_path, input_shape=(1, 3, 64, 64)):
    """
    モデルをONNX形式でエクスポート
    
    Parameters:
    -----------
    model : nn.Module
        エクスポートするモデル
    save_path : str
        保存先のパス
    input_shape : tuple
        入力テンソルの形状
    """
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"ONNX model exported to {save_path}")
```
