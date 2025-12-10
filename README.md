# TsuruTune 2.0 - Edge Device Deep Learning Optimizer

**[English](#english) | [日本語](#日本語-japanese)**

---

<a name="english"></a>
## English

TsuruTune is a comprehensive deep learning model optimization tool designed for edge devices and embedded platforms. It leverages hardware acceleration (Tensor Cores, CUDA) and memory bandwidth alignment to achieve optimal performance for deep learning inference on resource-constrained devices.

## Features

### Model Optimization
- **TensorRT Integration**: Full TensorRT optimization with CUDA support
- **ONNX Runtime**: Comprehensive CPU optimization with quantization
- **Multiple Precision Formats**: FP32, FP16, BF16, INT8 support
- **Advanced Quantization**: Per-channel, symmetric, and KV-cache quantization
- **Pruning & Sparsity**: Structured and unstructured pruning patterns
- **Graph Optimizations**: Batch normalization folding, constant folding, graph fusion

### User Interface
- **Modern Electron App**: Cross-platform desktop application
- **Intuitive Dashboard**: Real-time optimization statistics and trends
- **History Management**: Complete optimization history with parameter tracking
- **Device Configuration**: Separate optimization panels for CUDA and CPU
- **Progress Tracking**: Real-time optimization progress visualization
- **Batch Optimization**: Generate multiple optimized models with different parameter combinations

### Advanced Features
- **Local Model Storage**: Organized model management with metadata
- **Optimization History**: Persistent history with rerun capabilities
- **Performance Analytics**: Detailed performance gain and memory reduction metrics with real benchmarking
- **Export Capabilities**: Save optimized models to any location, generate detailed reports, history export in JSON and CSV formats
- **GitHub Integration**: Direct access to project repository
- **16 CPU Optimization Parameters**: Complete control over quantization, pruning, graph optimizations, and runtime configuration

## Requirements

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Node.js**: Version 16.0 or higher
- **Python**: Version 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended

### For CUDA Optimization (Optional)
- **NVIDIA GPU**: CUDA-compatible GPU (NVIDIA Jetson, RTX, etc.)
- **CUDA Toolkit**: Version 11.0 or higher (JetPack 5.0+ for Jetson)
- **TensorRT**: Version 8.5 or higher
- **PyTorch**: Version 2.0 or higher (use NVIDIA wheels for Jetson)

### For CPU Optimization
- **ONNX Runtime**: Automatically installed
- **NumPy**: Automatically installed

## Installation

### Quick Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/fsudjatmiko/tsurutune-app.git
   cd tsurutune-app
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Setup Python environment:**
   ```bash
   # On macOS/Linux
   ./setup.sh
   
   # On Windows
   setup.bat
   ```

4. **Start the application:**
   ```bash
   npm start
   ```

### Manual Python Setup
If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r python/requirements.txt

# For CUDA support on desktop GPUs
pip install torch torchvision tensorrt

# For Jetson devices (Orin, Xavier, Nano)
# Use NVIDIA-provided PyTorch wheels from:
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# TensorRT is pre-installed with JetPack
```

### Jetson-Specific Setup (Orin Nano, Xavier, etc.)

For NVIDIA Jetson devices with JetPack 5.0+:

```bash
# Install Node.js 18 for ARM64
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Clone and setup
git clone https://github.com/fsudjatmiko/tsurutune-app.git
cd tsurutune-app
npm install

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CPU dependencies
pip install numpy onnx onnxruntime psutil

# Install PyTorch for Jetson (download appropriate wheel)
# Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# Example for JetPack 5.1:
wget https://nvidia.box.com/shared/static/[pytorch-wheel-url].whl
pip install torch-*.whl

# Install TensorRT Python bindings (TensorRT already in JetPack)
pip install pycuda

# Optional: TensorFlow for Keras models
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow

# Start the application
npm start
```

## 📖 Usage Guide

### 1. Model Import
- Click "Add New Model" on the dashboard
- Select your ONNX, PyTorch (.pt/.pth), or TensorFlow (.pb) model
- The model will be imported into local storage

### 2. Optimization Configuration

#### CUDA/GPU Optimization
- **Precision**: Choose from FP32, FP16, BF16, or INT8
- **Quantization**: Configure per-channel and symmetric quantization
- **Calibration**: Provide calibration dataset for INT8
- **Pruning**: Set sparsity patterns and targets
- **Engine Settings**: Configure batch size, workspace, and tactics

#### CPU Optimization
- **Precision**: FP32, FP16, BF16, or INT8 quantization
- **Graph Optimizations**: Enable fusion, folding, batch normalization merging
- **Threading**: Configure intra-op and inter-op thread counts for optimal performance
- **Pruning**: Channel pruning, clustering, and sparsity patterns
- **Calibration**: Configurable calibration samples for accurate quantization
- **Runtime Configuration**: Batch size, execution providers, and optimization levels

### 3. Running Optimization
1. Navigate to the "Optimize" page
2. Select your target device (CUDA or CPU)
3. Configure optimization parameters
4. Click "Start Optimization"
5. Monitor real-time progress

### 4. History Management
- View all optimization attempts in the "History" page
- Filter by device, status, or date
- View detailed parameters for each optimization
- Rerun successful optimizations with the same settings
- Export history for analysis

### 5. Batch Optimization
- Navigate to the "Batch Optimize" page
- Select a model from your library
- Choose optimization variants:
  - **Precision Formats**: FP32, FP16, BF16, INT8
  - **Graph Optimizations**: Enabled/Disabled
  - **Pruning Options**: None/Light/Aggressive
- Use quick presets (All, Recommended) or custom combinations
- Start batch optimization to generate multiple optimized models
- Compare results to find the best configuration

### 6. Export & Save Models
- After optimization, click "Save to Library" to save the optimized model to any location
- Click "Generate Report" to create a detailed optimization report with metrics
- Use the file explorer dialog to choose save location

### 7. Analytics Dashboard
The dashboard provides:
- **Model Statistics**: Total models and optimizations
- **Performance Metrics**: Average gains and memory reduction with real benchmarking
- **Success Rates**: Optimization success statistics
- **Activity Feed**: Recent optimization activities
- **Device Usage**: Most used devices and precision formats

## Architecture

### Frontend (Electron)
```
src/
├── main/           # Electron main process
│   ├── main.js     # Application entry point
│   └── preload.js  # IPC bridge
└── renderer/       # UI components
    ├── index.html  # Main interface
    ├── renderer.js # Frontend logic
    └── css/        # Styling
```

### Backend (Python)
```
python/
├── main.py              # Backend entry point
├── model_manager.py     # Model storage management
├── history_manager.py   # Optimization history
├── optimizers/
│   ├── cuda_optimizer.py   # TensorRT optimization
│   └── cpu_optimizer.py    # ONNX Runtime optimization
└── utils/
    └── logger.py        # Logging utilities
```

### Communication Flow
1. **Frontend** → Electron IPC → **Main Process**
2. **Main Process** → Python subprocess → **Backend**
3. **Backend** → JSON response → **Main Process**
4. **Main Process** → IPC response → **Frontend**

## 🔧 Development

### Running in Development Mode
```bash
npm run dev
```

### Building for Production
```bash
# Build for current platform
npm run build

# Build for specific platforms
npm run build:win    # Windows
npm run build:mac    # macOS
npm run build:linux  # Linux
```

### Python Backend Testing
```bash
# Test system information
python python/main.py system

# Test optimization history
python python/main.py history

# Test with configuration
python python/main.py optimize --config '{"modelPath":"/path/to/model.onnx","device":"cpu"}'
```

## Performance Benchmarks

Typical optimization results on edge devices:

| Model Type | Original Size | Optimized Size | Performance Gain | Memory Reduction |
|------------|---------------|----------------|------------------|------------------|
| ResNet-50  | 98MB         | 25MB (INT8)   | +45%            | 74%             |
| ResNet-50  | 98MB         | 49MB (FP16)   | +30%            | 50%             |
| YOLOv5     | 45MB         | 12MB (INT8)   | +60%            | 73%             |
| BERT-Base  | 110MB        | 28MB (INT8)   | +35%            | 75%             |

*Results may vary based on hardware configuration and optimization settings. Benchmarks performed using real inference timing.*

## Testing

### Running Tests
```bash
# Frontend tests
npm test

# Python backend tests
python -m pytest python/tests/

# Integration tests
npm run test:integration
```

### Code Style
- **JavaScript**: ESLint configuration included
- **Python**: Follow PEP 8 guidelines
- **Commits**: Use conventional commit messages

## Acknowledgments

- **NVIDIA** for TensorRT and CUDA technologies
- **Microsoft** for ONNX Runtime
- **Electron** for the cross-platform framework
- **Open Source Community** for various libraries and tools

## Roadmap

### Version 2.0 (Planned)
- [ ] Multi-GPU optimization support
- [ ] Custom optimization profiles
- [ ] Model comparison tools
- [ ] Cloud deployment integration
- [ ] Advanced pruning algorithms

### Version 1.1 (Current)
- [x] Complete TensorRT integration
- [x] ONNX Runtime optimization with all 16 parameters
- [x] History management system
- [x] Performance analytics with real benchmarking
- [x] Batch optimization with preset combinations
- [x] FP16/BF16 CPU optimization support
- [x] Export and save optimized models
- [x] Detailed optimization reports
- [ ] Model validation tools
- [ ] Advanced pruning algorithms

---

<a name="日本語-japanese"></a>
## 日本語 (Japanese)

TsuruTuneは、エッジデバイスと組み込みプラットフォーム向けに設計された包括的な深層学習モデル最適化ツールです。ハードウェアアクセラレーション（Tensor Core、CUDA）とメモリ帯域幅アライメントを活用して、リソース制約のあるデバイスでの深層学習推論の最適なパフォーマンスを実現します。

## 機能

### モデル最適化
- **TensorRT統合**: CUDAサポートによる完全なTensorRT最適化
- **ONNX Runtime**: 量子化を含む包括的なCPU最適化
- **複数精度形式**: FP32、FP16、BF16、INT8サポート
- **高度な量子化**: チャネル毎、対称、KVキャッシュ量子化
- **プルーニング＆スパース化**: 構造化・非構造化プルーニングパターン
- **グラフ最適化**: バッチ正規化畳み込み、定数畳み込み、グラフ融合

### ユーザーインターフェース
- **モダンElectronアプリ**: クロスプラットフォームデスクトップアプリケーション
- **直感的なダッシュボード**: リアルタイム最適化統計とトレンド
- **履歴管理**: パラメータ追跡による完全な最適化履歴
- **デバイス設定**: CUDAとCPU用の個別最適化パネル
- **進捗追跡**: リアルタイム最適化進捗可視化
- **バッチ最適化**: 異なるパラメータ組み合わせで複数の最適化モデルを生成

### 高度な機能
- **ローカルモデルストレージ**: メタデータ付き整理されたモデル管理
- **最適化履歴**: 再実行機能付き永続履歴
- **パフォーマンス分析**: 実際のベンチマークによる詳細なパフォーマンス向上とメモリ削減メトリクス
- **エクスポート機能**: 最適化モデルを任意の場所に保存、詳細レポート生成、JSON・CSV形式での履歴エクスポート
- **GitHub統合**: プロジェクトリポジトリへの直接アクセス
- **16のCPU最適化パラメータ**: 量子化、プルーニング、グラフ最適化、ランタイム設定の完全制御

## 動作要件

### システム要件
- **オペレーティングシステム**: Windows 10+、macOS 10.14+、Ubuntu 18.04+
- **Node.js**: バージョン16.0以上
- **Python**: バージョン3.8以上
- **メモリ**: 最小4GB RAM、推奨8GB

### CUDA最適化用（オプション）
- **NVIDIA GPU**: CUDA対応GPU
- **CUDA Toolkit**: バージョン11.0以上
- **TensorRT**: バージョン8.6以上
- **PyTorch**: バージョン2.0以上

### CPU最適化用
- **ONNX Runtime**: 自動インストール
- **NumPy**: 自動インストール

## インストール

### クイックセットアップ
1. **リポジトリをクローン:**
   ```bash
   git clone https://github.com/fsudjatmiko/tsurutune-app.git
   cd tsurutune-app
   ```

2. **Node.js依存関係をインストール:**
   ```bash
   npm install
   ```

3. **Python環境をセットアップ:**
   ```bash
   # macOS/Linux
   ./setup.sh
   
   # Windows
   setup.bat
   ```

4. **アプリケーションを起動:**
   ```bash
   npm start
   ```

### 手動Python設定
手動設定を希望する場合:

```bash
# 仮想環境を作成
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係をインストール
pip install -r python/requirements.txt

# CUDAサポート用（オプション）
pip install torch torchvision tensorrt
```

## 📖 使用方法

### 1. モデルインポート
- ダッシュボードで「新しいモデルを追加」をクリック
- ONNX、PyTorch（.pt/.pth）、またはTensorFlow（.pb）モデルを選択
- モデルがローカルストレージにインポートされます

### 2. 最適化設定

#### CUDA/GPU最適化
- **精度**: FP32、FP16、BF16、またはINT8から選択
- **量子化**: チャネル毎および対称量子化を設定
- **キャリブレーション**: INT8用キャリブレーションデータセットを提供
- **プルーニング**: スパース化パターンとターゲットを設定
- **エンジン設定**: バッチサイズ、ワークスペース、戦術を設定

#### CPU最適化
- **精度**: FP32、FP16、BF16、またはINT8量子化
- **グラフ最適化**: 融合、畳み込み、バッチ正規化統合を有効化
- **スレッド**: 最適なパフォーマンスのためのintra-opおよびinter-opスレッド数設定
- **プルーニング**: チャネルプルーニング、クラスタリング、スパース化パターン
- **キャリブレーション**: 正確な量子化のための設定可能なキャリブレーションサンプル数
- **ランタイム設定**: バッチサイズ、実行プロバイダー、最適化レベル

### 3. 最適化実行
1. 「最適化」ページに移動
2. ターゲットデバイス（CUDAまたはCPU）を選択
3. 最適化パラメータを設定
4. 「最適化開始」をクリック
5. リアルタイム進捗を監視

### 4. 履歴管理
- 「履歴」ページですべての最適化試行を表示
- デバイス、ステータス、または日付でフィルタ
- 各最適化の詳細パラメータを表示
- 同じ設定で成功した最適化を再実行
- 分析用履歴エクスポート

### 5. バッチ最適化
- 「バッチ最適化」ページに移動
- ライブラリからモデルを選択
- 最適化バリアントを選択:
  - **精度形式**: FP32、FP16、BF16、INT8
  - **グラフ最適化**: 有効/無効
  - **プルーニングオプション**: なし/軽量/積極的
- クイックプリセット（すべて、推奨）またはカスタム組み合わせを使用
- バッチ最適化を開始して複数の最適化モデルを生成
- 結果を比較して最適な設定を見つける

### 6. モデルのエクスポートと保存
- 最適化後、「ライブラリに保存」をクリックして最適化モデルを任意の場所に保存
- 「レポート生成」をクリックしてメトリクス付き詳細最適化レポートを作成
- ファイルエクスプローラーダイアログで保存場所を選択

### 7. 分析ダッシュボード
ダッシュボードでは以下を提供:
- **モデル統計**: 総モデル数と最適化数
- **パフォーマンスメトリクス**: 実際のベンチマークによる平均向上とメモリ削減
- **成功率**: 最適化成功統計
- **アクティビティフィード**: 最近の最適化活動
- **デバイス使用**: 最も使用されるデバイスと精度形式

## アーキテクチャ

### フロントエンド（Electron）
```
src/
├── main/           # Electronメインプロセス
│   ├── main.js     # アプリケーションエントリポイント
│   └── preload.js  # IPCブリッジ
└── renderer/       # UIコンポーネント
    ├── index.html  # メインインターフェース
    ├── renderer.js # フロントエンドロジック
    └── css/        # スタイリング
```

### バックエンド（Python）
```
python/
├── main.py              # バックエンドエントリポイント
├── model_manager.py     # モデルストレージ管理
├── history_manager.py   # 最適化履歴
├── optimizers/
│   ├── cuda_optimizer.py   # TensorRT最適化
│   └── cpu_optimizer.py    # ONNX Runtime最適化
└── utils/
    └── logger.py        # ログユーティリティ
```

### 通信フロー
1. **フロントエンド** → Electron IPC → **メインプロセス**
2. **メインプロセス** → Pythonサブプロセス → **バックエンド**
3. **バックエンド** → JSON応答 → **メインプロセス**
4. **メインプロセス** → IPC応答 → **フロントエンド**

## 🔧 開発

### 開発モードで実行
```bash
npm run dev
```

### プロダクション用ビルド
```bash
# 現在のプラットフォーム用ビルド
npm run build

# 特定プラットフォーム用ビルド
npm run build:win    # Windows
npm run build:mac    # macOS
npm run build:linux  # Linux
```

### Pythonバックエンドテスト
```bash
# システム情報テスト
python python/main.py system

# 最適化履歴テスト
python python/main.py history

# 設定付きテスト
python python/main.py optimize --config '{"modelPath":"/path/to/model.onnx","device":"cpu"}'
```

## パフォーマンスベンチマーク

エッジデバイスでの典型的な最適化結果:

| モデルタイプ | 元のサイズ | 最適化後サイズ | パフォーマンス向上 | メモリ削減 |
|-------------|-----------|---------------|------------------|----------|
| ResNet-50   | 98MB      | 25MB (INT8)   | +45%             | 74%      |
| ResNet-50   | 98MB      | 49MB (FP16)   | +30%             | 50%      |
| YOLOv5      | 45MB      | 12MB (INT8)   | +60%             | 73%      |
| BERT-Base   | 110MB     | 28MB (INT8)   | +35%             | 75%      |

*結果はハードウェア構成と最適化設定により異なる場合があります。実際の推論タイミングを使用してベンチマークを実行。*

## テスト

### テスト実行
```bash
# フロントエンドテスト
npm test

# Pythonバックエンドテスト
python -m pytest python/tests/

# 統合テスト
npm run test:integration
```

### コードスタイル
- **JavaScript**: ESLint設定を含む
- **Python**: PEP 8ガイドラインに従う
- **Commits**: 従来のコミットメッセージを使用

## 謝辞

- **NVIDIA** - TensorRTとCUDA技術
- **Microsoft** - ONNX Runtime
- **Electron** - クロスプラットフォームフレームワーク
- **Open Source Community** - 各種ライブラリとツール

## ロードマップ

### バージョン2.0（予定）
- [ ] マルチGPU最適化サポート
- [ ] カスタム最適化プロファイル
- [ ] モデル比較ツール
- [ ] クラウドデプロイ統合
- [ ] 高度なプルーニングアルゴリズム

### バージョン1.1（現在）
- [x] 完全なTensorRT統合
- [x] 全16パラメータを使用したONNX Runtime最適化
- [x] 履歴管理システム
- [x] 実際のベンチマークによるパフォーマンス分析
- [x] プリセット組み合わせによるバッチ最適化
- [x] FP16/BF16 CPU最適化サポート
- [x] 最適化モデルのエクスポートと保存
- [x] 詳細な最適化レポート
- [ ] モデル検証ツール
- [ ] 高度なプルーニングアルゴリズム

---
*Developed by Farrell Rafee Sudjatmiko - ITS Computer Engineering*
