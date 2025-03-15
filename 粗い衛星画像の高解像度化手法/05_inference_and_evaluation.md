# 衛星画像超解像システム - 推論と評価

## 1. 推論プロセスの概要

推論プロセスは、学習済みのSwin2SRモデルを使用して、低解像度の衛星画像から高解像度の画像を生成するプロセスです。このモジュールでは、モデルの読み込み、画像の前処理、推論の実行、後処理、および結果の評価と可視化を行います。

## 2. 推論パイプライン

### 2.1 モデルの読み込み

```python
import torch
import os

def load_model(model_path, device='cuda'):
    """
    学習済みモデルの読み込み
    
    Parameters:
    -----------
    model_path : str
        モデルファイルのパス
    device : str
        使用するデバイス ('cuda' or 'cpu')
        
    Returns:
    --------
    nn.Module
        読み込まれたモデル
    """
    # モデルの初期化
    model = Swin2SR(
        upscale=4,
        img_size=64,
        window_size=8,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        upsampler='pixelshuffle'
    )
    
    # 学習済みの重みを読み込み
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 評価モードに設定
    model.eval()
    
    # デバイスに転送
    model = model.to(device)
    
    return model
```

### 2.2 画像の前処理

```python
import cv2
import numpy as np
import torch

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
```

### 2.3 推論の実行

```python
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
```

### 2.4 後処理

```python
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
```

### 2.5 完全な推論パイプライン

```python
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
    model = load_model(model_path, device)
    
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

## 3. バッチ処理

複数の画像を一括で処理するためのバッチ処理機能です。

```python
def batch_super_resolve(model_path, input_dir, output_dir, device='cuda'):
    """
    ディレクトリ内の複数画像に対して超解像処理を行う
    
    Parameters:
    -----------
    model_path : str
        モデルファイルのパス
    input_dir : str
        入力画像のディレクトリ
    output_dir : str
        出力画像の保存先ディレクトリ
    device : str
        使用するデバイス ('cuda' or 'cpu')
    """
    # モデルの読み込み
    model = load_model(model_path, device)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 入力ディレクトリ内の画像ファイルを取得
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"SR_{image_file}")
        
        # 画像の前処理
        lr_tensor, original_size = preprocess_image(input_path, device)
        
        # 推論
        sr_tensor = inference(model, lr_tensor, device)
        
        # 後処理
        sr_img = postprocess_image(sr_tensor, original_size)
        
        # 結果の保存
        cv2.imwrite(output_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
        
        print(f"Processed: {image_file} -> {output_path}")
```

## 4. 評価指標

### 4.1 PSNR (Peak Signal-to-Noise Ratio)

```python
from skimage.metrics import peak_signal_noise_ratio

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
        sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 値の範囲を0-1に正規化
    sr_img_norm = sr_img.astype(np.float32) / 255.0
    hr_img_norm = hr_img.astype(np.float32) / 255.0
    
    # PSNR計算
    return peak_signal_noise_ratio(hr_img_norm, sr_img_norm, data_range=1.0)
```

### 4.2 SSIM (Structural Similarity Index)

```python
from skimage.metrics import structural_similarity

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
        sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 値の範囲を0-1に正規化
    sr_img_norm = sr_img.astype(np.float32) / 255.0
    hr_img_norm = hr_img.astype(np.float32) / 255.0
    
    # SSIM計算
    return structural_similarity(hr_img_norm, sr_img_norm, data_range=1.0, multichannel=True)
```

### 4.3 評価の実行

```python
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

## 5. 結果の可視化

### 5.1 比較画像の生成

```python
def create_comparison_image(lr_img, sr_img, hr_img=None):
    """
    低解像度、超解像、高解像度（オプション）の比較画像を生成
    
    Parameters:
    -----------
    lr_img : np.ndarray
        低解像度画像
    sr_img : np.ndarray
        超解像画像
    hr_img : np.ndarray, optional
        高解像度参照画像
        
    Returns:
    --------
    np.ndarray
        比較画像
    """
    # 低解像度画像を超解像画像のサイズにリサイズ
    lr_img_resized = cv2.resize(lr_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    if hr_img is not None:
        # 高解像度画像が提供されている場合は3画像を並べる
        # 高解像度画像のサイズが異なる場合はリサイズ
        if hr_img.shape != sr_img.shape:
            hr_img = cv2.resize(hr_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # 3画像を水平に連結
        comparison = np.concatenate((lr_img_resized, sr_img, hr_img), axis=1)
        
        # ラベルを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'LR (Bicubic)', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comparison, 'SR (Swin2SR)', (sr_img.shape[1] + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comparison, 'HR (Ground Truth)', (2 * sr_img.shape[1] + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # 高解像度画像がない場合は2画像を並べる
        comparison = np.concatenate((lr_img_resized, sr_img), axis=1)
        
        # ラベルを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'LR (Bicubic)', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comparison, 'SR (Swin2SR)', (sr_img.shape[1] + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return comparison
```

### 5.2 拡大表示

```python
def create_zoom_comparison(lr_img, sr_img, hr_img=None, zoom_factor=4, region=None):
    """
    特定領域を拡大した比較画像を生成
    
    Parameters:
    -----------
    lr_img : np.ndarray
        低解像度画像
    sr_img : np.ndarray
        超解像画像
    hr_img : np.ndarray, optional
        高解像度参照画像
    zoom_factor : int
        拡大倍率
    region : tuple, optional
        拡大する領域 (x, y, width, height)
        
    Returns:
    --------
    np.ndarray
        拡大比較画像
    """
    # 低解像度画像を超解像画像のサイズにリサイズ
    lr_img_resized = cv2.resize(lr_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 拡大する領域が指定されていない場合は中央部分を使用
    if region is None:
        h, w = sr_img.shape[:2]
        x = w // 4
        y = h // 4
        width = w // 2
        height = h // 2
        region = (x, y, width, height)
    
    x, y, width, height = region
    
    # 領域の切り出し
    lr_crop = lr_img_resized[y:y+height, x:x+width]
    sr_crop = sr_img[y:y+height, x:x+width]
    
    # 拡大
    lr_zoom = cv2.resize(lr_crop, (width * zoom_factor, height * zoom_factor), interpolation=cv2.INTER_NEAREST)
    sr_zoom = cv2.resize(sr_crop, (width * zoom_factor, height * zoom_factor), interpolation=cv2.INTER_NEAREST)
    
    if hr_img is not None:
        # 高解像度画像が提供されている場合
        if hr_img.shape != sr_img.shape:
            hr_img = cv2.resize(hr_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        hr_crop = hr_img[y:y+height, x:x+width]
        hr_zoom = cv2.resize(hr_crop, (width * zoom_factor, height * zoom_factor), interpolation=cv2.INTER_NEAREST)
        
        # 3画像を水平に連結
        zoom_comparison = np.concatenate((lr_zoom, sr_zoom, hr_zoom), axis=1)
        
        # 元の画像に拡大領域を表示
        lr_with_rect = lr_img_resized.copy()
        sr_with_rect = sr_img.copy()
        hr_with_rect = hr_img.copy()
        
        cv2.rectangle(lr_with_rect, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.rectangle(sr_with_rect, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.rectangle(hr_with_rect, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # 元の画像を水平に連結
        original_comparison = np.concatenate((lr_with_rect, sr_with_rect, hr_with_rect), axis=1)
        
        # 元の画像と拡大画像を垂直に連結
        final_comparison = np.concatenate((original_comparison, zoom_comparison), axis=0)
    else:
        # 高解像度画像がない場合
        zoom_comparison = np.concatenate((lr_zoom, sr_zoom), axis=1)
        
        # 元の画像に拡大領域を表示
        lr_with_rect = lr_img_resized.copy()
        sr_with_rect = sr_img.copy()
        
        cv2.rectangle(lr_with_rect, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.rectangle(sr_with_rect, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # 元の画像を水平に連結
        original_comparison = np.concatenate((lr_with_rect, sr_with_rect), axis=1)
        
        # 元の画像と拡大画像を垂直に連結
        final_comparison = np.concatenate((original_comparison, zoom_comparison), axis=0)
    
    return final_comparison
```

### 5.3 評価結果のプロット

```python
import matplotlib.pyplot as plt

def plot_evaluation_results(results, save_path=None):
    """
    評価結果をプロットする
    
    Parameters:
    -----------
    results : dict
        評価結果
    save_path : str, optional
        保存先のパス
    """
    # ファイル名と評価指標の取得
    file_names = results['file_names']
    psnr_values = results['psnr']
    ssim_values = results['ssim']
    
    # プロットの作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # PSNR
    ax1.bar(file_names, psnr_values, color='skyblue')
    ax1.set_title('PSNR Values')
    ax1.set_xlabel('Image')
    ax1.set_ylabel('PSNR (dB)')
    ax1.axhline(y=results['avg_psnr'], color='r', linestyle='-', label=f'Average: {results["avg_psnr"]:.2f} dB')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=90)
    
    # SSIM
    ax2.bar(file_names, ssim_values, color='lightgreen')
    ax2.set_title('SSIM Values')
    ax2.set_xlabel('Image')
    ax2.set_ylabel('SSIM')
    ax2.axhline(y=results['avg_ssim'], color='r', linestyle='-', label=f'Average: {results["avg_ssim"]:.4f}')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Evaluation plot saved to {save_path}")
    
    plt.show()
```

## 6. 推論スクリプト

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Super-resolution inference for satellite images')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output image or directory')
    parser.add_argument('--reference', type=str, default=None, help='Path to high-resolution reference image or directory')
    parser.add_argument('--batch', action='store_true', help='Process a batch of images')
    parser.add_argument('--compare', action='store_true', help='Create comparison images')
    parser.add_argument('--zoom', action='store_true', help='Create zoomed comparison images')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate results using reference images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # デバイスの確認
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        device = 'cpu'
    
    # バッチ処理
    if args.batch:
        if not os.path.isdir(args.input):
            print("Error: Input must be a directory when using batch processing.")
            return
        
        os.makedirs(args.output, exist_ok=True)
        
        # バッチ超解像処理
        batch_super_resolve(args.model, args.input, args.output, device)
        
        # 評価
        if args.evaluate and args.reference:
            if not os.path.isdir(args.reference):
                print("Error: Reference must be a directory when evaluating batch results.")
                return
            
            results = evaluate_super_resolution(args.output, args.reference)
            
            # 評価結果のプロット
            plot_path = os.path.join(args.output, 'evaluation_results.png')
            plot_evaluation_results(results, save_path=plot_path)
    else:
        # 単一画像処理
        if not os.path.isfile(args.input):
            print("Error: Input must be a file when not using batch processing.")
            return
        
        # 超解像処理
        sr_img = super_resolve_image(args.model, args.input, args.output, device)
        
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

if __name__ == '__main__':
    main()
```

## 7. 実行例

### 7.1 単一画像の超解像

```bash
python inference.py --model ./models/best_model.pth --input ./data/test/lr/image.png --output ./results/sr_image.png
```

### 7.2 比較画像の作成

```bash
python inference.py --model ./models/best_model.pth --input ./data/test/lr/image.png --output ./results/sr_image.png --reference ./data/test/hr/image.png --compare --zoom
```

### 7.3 バッチ処理と評価

```bash
python inference.py --model ./models/best_model.pth --input ./data/test/lr --output ./results/sr --reference ./data/test/hr --batch --evaluate
```

## 8. 推論の最適化

### 8.1 TorchScript変換

```python
def convert_to_torchscript(model, example_input, save_path):
    """
    モデルをTorchScriptに変換
    
    Parameters:
    -----------
    model : nn.Module
        変換するモデル
    example_input : torch.Tensor
        サンプル入力
    save_path : str
        保存先のパス
    """
    model.eval()
    
    # トレース
    traced_model = torch.jit.trace(model, example_input)
    
    # 保存
    torch.jit.save(traced_model, save_path)
    print(f"TorchScript model saved to {save_path}")
    
    return traced_model
```

### 8.2 ONNX変換

```python
def convert_to_onnx(model, example_input, save_path):
    """
    モデルをONNXに変換
    
    Parameters:
    -----------
    model : nn.Module
        変換するモデル
    example_input : torch.Tensor
        サンプル入力
    save_path : str
        保存先のパス
    """
    model.eval()
    
    # ONNX変換
    torch.onnx.export(
        model,
        example_input,
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
    
    print(f"ONNX model saved to {save_path}")
```

### 8.3 TensorRT変換

```python
def convert_to_tensorrt(onnx_path, save_path):
    """
    ONNXモデルをTensorRTに変換
    
    Parameters:
    -----------
    onnx_path : str
        ONNXモデルのパス
    save_path : str
        保存先のパス
    """
    import tensorrt as trt
    
    # TensorRTロガーの作成
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # ビルダーの作成
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # ONNXモデルの解析
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 設定
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # エンジンの構築
    engine = builder.build_engine(network, config)
    
    # 保存
    with open(save_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved to {save_path}")
    
    return engine
```

### 8.4 最適化モデルを使用した推論

```python
def inference_with_torchscript(model_path, lr_tensor, device='cuda'):
    """
    TorchScriptモデルを使用した推論
    
    Parameters:
    -----------
    model_path : str
        TorchScriptモデルのパス
    lr_tensor : torch.Tensor
        低解像度画像テンソル
    device : str
        使用するデバイス
        
    Returns:
    --------
    torch.Tensor
        超解像画像テンソル
    """
    # モデルの読み込み
    model = torch.jit.load(model_path, map_location=device)
    
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    return sr_tensor
```
