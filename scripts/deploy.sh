#!/bin/bash
# LLM-HPC 服务器部署脚本

set -e

echo "=========================================="
echo "  LLM-HPC 部署脚本"
echo "=========================================="

# 检查 Python 版本
python3 --version || { echo "Python3 not found"; exit 1; }

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 创建必要目录
echo "Creating directories..."
mkdir -p logs results/analysis results/conversion benchmarks/minimd benchmarks/hpcg

# 检查配置文件
if [ ! -f "configs/config.yaml" ]; then
    echo "Copying config template..."
    cp configs/config.example.yaml configs/config.yaml 2>/dev/null || true
fi

# 检查 API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key'"
    echo "Or add it to .env file"
fi

# 检查 CUDA (可选)
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA found: $(nvcc --version | grep release)"
else
    echo "⚠️  CUDA not found (optional, needed for GPU conversion testing)"
fi

echo ""
echo "=========================================="
echo "  部署完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  source venv/bin/activate"
echo "  python src/main.py --help"
echo ""
echo "快速开始:"
echo "  python src/main.py analyze --code minimd --prompt all"
echo "  python src/main.py convert --code minimd --function compute_fullneigh"
echo ""
