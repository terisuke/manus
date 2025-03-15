# 衛星画像超解像システム - 学習プロセス

## 1. 学習プロセスの概要

学習プロセスは、データパイプラインから提供されるデータを使用して、Swin2SRモデルを効率的に学習させるためのプロセスです。このモジュールでは、損失関数、最適化手法、学習スケジュール、評価指標などを定義し、モデルの学習を管理します。

## 2. 損失関数

### 2.1 SSIM損失関数

構造的類似性指標（SSIM）に基づく損失関数を使用します。SSIMは人間の視覚システムに近い評価を行うため、超解像タスクに適しています。

```python
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
```

### 2.2 複合損失関数

SSIM損失とL1損失（平均絶対誤差）を組み合わせた複合損失関数も効果的です。

```python
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

## 3. 最適化手法

### 3.1 オプティマイザ

AdamWオプティマイザを使用します。これは、重み減衰を適切に適用するAdam拡張版です。

```python
import torch.optim as optim

def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """
    モデルのオプティマイザを取得
    
    Parameters:
    -----------
    model : nn.Module
        最適化するモデル
    lr : float
        学習率
    weight_decay : float
        重み減衰係数
        
    Returns:
    --------
    torch.optim.Optimizer
        オプティマイザ
    """
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

### 3.2 学習率スケジューラ

コサインアニーリングスケジューラを使用して、学習率を徐々に減少させます。

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_scheduler(optimizer, num_epochs, eta_min=1e-7):
    """
    学習率スケジューラを取得
    
    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        オプティマイザ
    num_epochs : int
        エポック数
    eta_min : float
        最小学習率
        
    Returns:
    --------
    torch.optim.lr_scheduler._LRScheduler
        学習率スケジューラ
    """
    return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
```

## 4. 学習ループ

### 4.1 トレーニングループ

```python
import time
import os
from tqdm import tqdm
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

### 4.2 混合精度学習

```python
from torch.cuda.amp import autocast, GradScaler

def train_model_with_amp(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                         num_epochs=100, device='cuda', save_dir='./checkpoints', writer=None):
    """
    混合精度学習を使用したモデルの学習
    
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
    
    # 混合精度学習のためのスケーラー
    scaler = GradScaler()
    
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
            
            # 混合精度での順伝播
            with autocast():
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
            
            # スケーリングされた勾配の計算と更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
                
                # 混合精度での順伝播
                with autocast():
                    outputs = model(lr_imgs)
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

## 5. 評価指標

### 5.1 PSNR (Peak Signal-to-Noise Ratio)

```python
from skimage.metrics import peak_signal_noise_ratio

def calculate_psnr(img1, img2, data_range=1.0):
    """
    PSNR (Peak Signal-to-Noise Ratio) の計算
    
    Parameters:
    -----------
    img1, img2 : np.ndarray
        比較する画像
    data_range : float
        データの範囲
        
    Returns:
    --------
    float
        PSNR値 (dB)
    """
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)
```

### 5.2 SSIM (Structural Similarity Index)

```python
from skimage.metrics import structural_similarity

def calculate_ssim(img1, img2, data_range=1.0):
    """
    SSIM (Structural Similarity Index) の計算
    
    Parameters:
    -----------
    img1, img2 : np.ndarray
        比較する画像
    data_range : float
        データの範囲
        
    Returns:
    --------
    float
        SSIM値 (0-1)
    """
    return structural_similarity(img1, img2, data_range=data_range, multichannel=True)
```

## 6. 学習の可視化

### 6.1 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

def setup_tensorboard(log_dir):
    """
    TensorBoardのセットアップ
    
    Parameters:
    -----------
    log_dir : str
        ログディレクトリ
        
    Returns:
    --------
    SummaryWriter
        TensorBoardのSummaryWriter
    """
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)
```

### 6.2 学習曲線のプロット

```python
import matplotlib.pyplot as plt

def plot_learning_curves(history, save_path=None):
    """
    学習曲線のプロット
    
    Parameters:
    -----------
    history : dict
        学習履歴
    save_path : str, optional
        保存先のパス
    """
    # 損失のプロット
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_psnr'], label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('PSNR Curve')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_ssim'], label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('SSIM Curve')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to {save_path}")
    
    plt.show()
```

## 7. 学習実行スクリプト

```python
import argparse
import json
from datetime import datetime

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
    writer = setup_tensorboard(os.path.join(save_dir, 'logs'))
    
    # モデルの学習
    if args.amp:
        model, history = train_model_with_amp(
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
    else:
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
    parser.add_argument('--amp', action='store_true', help='Use mixed precision training')
    
    # 保存関連
    parser.add_argument('--save_dir', type=str, default='./experiments', help='Directory to save results')
    
    args = parser.parse_args()
    main(args)
```

## 8. 2段階学習戦略

2段階学習戦略を実装するためのスクリプトです。初期段階ではすべてのデータ拡張を適用し、後期段階ではCutMixを除外して微調整を行います。

```python
def two_stage_training(model, train_loader, val_loader, device='cuda', save_dir='./checkpoints'):
    """
    2段階学習戦略
    
    Parameters:
    -----------
    model : nn.Module
        学習するモデル
    train_loader : DataLoader
        訓練用データローダー
    val_loader : DataLoader
        検証用データローダー
    device : str
        使用するデバイス ('cuda' or 'cpu')
    save_dir : str
        モデルの保存ディレクトリ
        
    Returns:
    --------
    nn.Module
        学習済みモデル
    """
    # ステージ1: すべてのデータ拡張を適用し、比較的高い学習率で学習
    print("Stage 1: Training with all data augmentations")
    
    criterion = SSIMLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-6)
    
    stage1_dir = os.path.join(save_dir, 'stage1')
    os.makedirs(stage1_dir, exist_ok=True)
    
    writer1 = setup_tensorboard(os.path.join(stage1_dir, 'logs'))
    
    model, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=70,
        device=device,
        save_dir=stage1_dir,
        writer=writer1
    )
    
    # ステージ2: CutMixを除外し、低い学習率で微調整
    print("Stage 2: Fine-tuning without CutMix")
    
    # CutMixを除外した新しいデータローダーの作成
    train_loader_no_cutmix = get_dataloaders_no_cutmix(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )[0]
    
    # 最良のモデルを読み込み
    model.load_state_dict(torch.load(os.path.join(stage1_dir, 'best_model.pth')))
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)
    
    stage2_dir = os.path.join(save_dir, 'stage2')
    os.makedirs(stage2_dir, exist_ok=True)
    
    writer2 = setup_tensorboard(os.path.join(stage2_dir, 'logs'))
    
    model, _ = train_model(
        model=model,
        train_loader=train_loader_no_cutmix,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=30,
        device=device,
        save_dir=stage2_dir,
        writer=writer2
    )
    
    return model

def get_dataloaders_no_cutmix(data_dir, batch_size=16, num_workers=4):
    """
    CutMixを除外したデータローダーを取得
    
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
    # CutMixを除外したデータ拡張関数
    def transform_no_cutmix(image):
        # CutMixを除外した拡張処理
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
    
    # 訓練用データセット
    train_dataset = SatelliteDataset(
        lr_dir=f"{data_dir}/train/lr",
        hr_dir=f"{data_dir}/train/hr",
        transform=transform_no_cutmix,
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
