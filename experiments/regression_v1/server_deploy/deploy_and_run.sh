#!/bin/bash
# 服务器部署和运行脚本

set -e

echo "========================================================================"
echo "回归训练Pilot实验 - 服务器部署"
echo "========================================================================"

# 配置
SERVER_CODE="/root/regression-pilot-code"
SERVER_RESULTS="/root/regression-pilot-v1"
REPO_ROOT="/root/Fatigue-check-Mamba-v2"

# 步骤1: 创建代码目录
echo -e "\n[步骤1] 创建代码目录..."
mkdir -p "$SERVER_CODE"
cd "$SERVER_CODE"

# 步骤2: 上传文件（在本地执行）
echo -e "\n[步骤2] 请在本地执行以下命令上传文件："
echo "----------------------------------------"
echo "scp regression_models.py root@your-server:$SERVER_CODE/"
echo "scp regression_pilot.py root@your-server:$SERVER_CODE/"
echo "----------------------------------------"
echo "按Enter继续（上传完成后）..."
# read -p ""

# 步骤3: 检查文件
echo -e "\n[步骤3] 检查上传的文件..."
ls -lh "$SERVER_CODE"

# 步骤4: 检查环境
echo -e "\n[步骤4] 检查Python环境..."
python3 --version
echo -n "PyTorch: "
python3 -c "import torch; print(torch.__version__)"
echo -n "CUDA available: "
python3 -c "import torch; print(torch.cuda.is_available())"

# 步骤5: 检查数据
echo -e "\n[步骤5] 检查数据路径..."
DATA_PATH="/root/autodl-tmp/fatigue_mamba_data_v2/fatigue_time_axis_v2_preview"
if [ -d "$DATA_PATH" ]; then
    echo "✓ 数据路径存在: $DATA_PATH"
    ls -lh "$DATA_PATH/manifest.json"
else
    echo "✗ 数据路径不存在，请检查"
    exit 1
fi

# 步骤6: 检查代码库
echo -e "\n[步骤6] 检查主代码库..."
if [ -d "$REPO_ROOT" ]; then
    echo "✓ 代码库存在: $REPO_ROOT"
else
    echo "✗ 代码库不存在，请检查"
    exit 1
fi

# 步骤7: Dry run测试
echo -e "\n[步骤7] Dry run测试（不实际训练）..."
cd "$SERVER_CODE"
python3 regression_pilot.py \
    --variant C_E_regression \
    --fold 2 \
    --seed 17 \
    --max-steps 2304 \
    --output "$SERVER_RESULTS" \
    || { echo "Dry run失败"; exit 1; }

echo -e "\n✓ Dry run成功！"

# 步骤8: 询问是否开始训练
echo -e "\n========================================================================"
echo "准备开始实际训练"
echo "========================================================================"
echo "配置:"
echo "  模型: C_E_regression (EEG-only)"
echo "  Fold: 2"
echo "  步数: 2304"
echo "  输出: $SERVER_RESULTS"
echo ""
echo "预计时间: 1-2小时（取决于GPU）"
echo ""
read -p "确认开始训练? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "取消训练"
    exit 0
fi

# 步骤9: 开始训练
echo -e "\n[步骤9] 开始训练..."
echo "========================================================================"

cd "$SERVER_CODE"
python3 regression_pilot.py \
    --variant C_E_regression \
    --fold 2 \
    --seed 17 \
    --max-steps 2304 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 0.001 \
    --patience 10 \
    --output "$SERVER_RESULTS" \
    --execute \
    2>&1 | tee "$SERVER_RESULTS/training.log"

# 步骤10: 检查结果
echo -e "\n========================================================================"
echo "训练完成，检查结果"
echo "========================================================================"

if [ -f "$SERVER_RESULTS/final_metrics.json" ]; then
    echo -e "\n最终指标:"
    cat "$SERVER_RESULTS/final_metrics.json" | python3 -m json.tool

    echo -e "\n文件列表:"
    ls -lh "$SERVER_RESULTS"

    echo -e "\n✓ 训练成功完成！"
else
    echo -e "\n✗ 未找到结果文件，训练可能失败"
    exit 1
fi

echo -e "\n========================================================================"
echo "下一步："
echo "1. 下载结果: scp -r root@server:$SERVER_RESULTS ./results/"
echo "2. 分析指标，决定是否继续全面训练"
echo "========================================================================"
