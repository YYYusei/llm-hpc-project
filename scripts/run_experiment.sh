#!/bin/bash
# 批量运行实验脚本

set -e

# 配置
OUTPUT_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${OUTPUT_DIR}/experiment_${TIMESTAMP}"

echo "=========================================="
echo "  LLM-HPC 实验运行脚本"
echo "  实验目录: ${EXPERIMENT_DIR}"
echo "=========================================="

# 创建实验目录
mkdir -p "${EXPERIMENT_DIR}/analysis"
mkdir -p "${EXPERIMENT_DIR}/conversion"
mkdir -p "${EXPERIMENT_DIR}/logs"

# 激活虚拟环境
source venv/bin/activate

# 检查 API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

# ============== Phase 1: 性能分析实验 ==============
echo ""
echo "[Phase 1] 性能分析实验"
echo "========================================"

# miniMD 分析
echo "Analyzing miniMD..."
for prompt in zero_shot few_shot contextual; do
    echo "  - ${prompt}..."
    python src/main.py analyze \
        --code minimd \
        --prompt ${prompt} \
        --output "${EXPERIMENT_DIR}/analysis" \
        2>&1 | tee -a "${EXPERIMENT_DIR}/logs/analysis.log"
done

# HPCG 分析
echo "Analyzing HPCG..."
for prompt in zero_shot few_shot contextual; do
    echo "  - ${prompt}..."
    python src/main.py analyze \
        --code hpcg \
        --prompt ${prompt} \
        --output "${EXPERIMENT_DIR}/analysis" \
        2>&1 | tee -a "${EXPERIMENT_DIR}/logs/analysis.log"
done

# ============== Phase 2: GPU 转换实验 ==============
echo ""
echo "[Phase 2] GPU 转换实验"
echo "========================================"

# miniMD GPU 转换
echo "Converting miniMD compute_fullneigh..."
python src/main.py convert \
    --code minimd \
    --function compute_fullneigh \
    --output "${EXPERIMENT_DIR}/conversion" \
    2>&1 | tee -a "${EXPERIMENT_DIR}/logs/conversion.log"

# HPCG GPU 转换
echo "Converting HPCG ComputeSPMV_ref..."
python src/main.py convert \
    --code hpcg \
    --function ComputeSPMV_ref \
    --output "${EXPERIMENT_DIR}/conversion" \
    2>&1 | tee -a "${EXPERIMENT_DIR}/logs/conversion.log"

# ============== 生成报告 ==============
echo ""
echo "[Phase 3] 生成报告"
echo "========================================"

# 统计结果
ANALYSIS_COUNT=$(ls -1 "${EXPERIMENT_DIR}/analysis"/*.json 2>/dev/null | wc -l)
CONVERSION_COUNT=$(ls -1 "${EXPERIMENT_DIR}/conversion"/*.cu 2>/dev/null | wc -l)

echo "Analysis results: ${ANALYSIS_COUNT}"
echo "Conversion results: ${CONVERSION_COUNT}"

# 生成汇总
cat > "${EXPERIMENT_DIR}/summary.txt" << EOF
LLM-HPC 实验报告
================

实验时间: $(date)
实验目录: ${EXPERIMENT_DIR}

分析结果数量: ${ANALYSIS_COUNT}
转换结果数量: ${CONVERSION_COUNT}

文件列表:
---------
Analysis:
$(ls -1 "${EXPERIMENT_DIR}/analysis" 2>/dev/null || echo "  (none)")

Conversion:
$(ls -1 "${EXPERIMENT_DIR}/conversion" 2>/dev/null || echo "  (none)")
EOF

echo ""
echo "=========================================="
echo "  实验完成!"
echo "  结果目录: ${EXPERIMENT_DIR}"
echo "=========================================="
