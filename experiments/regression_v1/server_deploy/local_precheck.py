#!/usr/bin/env python3
"""
本地预检脚本 - 不需要PyTorch，只检查文件结构和逻辑

在服务器部署前运行，确保文件完整
"""
import sys
from pathlib import Path

def check_files():
    """检查文件是否齐全"""
    print("="*70)
    print("📋 检查部署文件")
    print("="*70)

    deploy_dir = Path(__file__).parent
    required_files = [
        "regression_models.py",
        "regression_pilot.py",
        "deploy_and_run.sh",
        "SERVER_GUIDE.md",
        "QUICK_START.md",
        "local_precheck.py"  # 本文件
    ]

    all_ok = True
    for fname in required_files:
        fpath = deploy_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"✅ {fname:30s} ({size:,} bytes)")
        else:
            print(f"❌ {fname:30s} 缺失!")
            all_ok = False

    return all_ok

def check_code_structure():
    """检查代码结构（不导入PyTorch模块）"""
    print("\n" + "="*70)
    print("🔍 检查代码结构")
    print("="*70)

    deploy_dir = Path(__file__).parent

    # 检查regression_models.py
    models_file = deploy_dir / "regression_models.py"
    with open(models_file, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = {
        "导入stream_seed": "from fatigue_training.development import stream_seed" in content,
        "导入DevelopmentModel": "from fatigue_training.development_models import DevelopmentModel" in content,
        "定义RegressionModalityModel": "class RegressionModalityModel" in content,
        "参数量检查": "COUNTS = {" in content,
        "BN1检查": "BN1 bias rejected" in content,
        "路径适配": "SERVER_CODE_ROOT" in content and "LOCAL_CODE_ROOT" in content
    }

    for name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    # 检查regression_pilot.py
    pilot_file = deploy_dir / "regression_pilot.py"
    with open(pilot_file, 'r', encoding='utf-8') as f:
        content = f.read()

    checks2 = {
        "导入configuration": "from fatigue_training.development import configuration" in content,
        "导入WindowDataset": "from fatigue_training.data import WindowDataset" in content,
        "导入normalization": "from fatigue_training.normalization import" in content,
        "定义RegressionTrainer": "class RegressionTrainer" in content,
        "计算指标": "def compute_metrics" in content,
        "路径适配": "SERVER_ROOT" in content and "LOCAL_ROOT" in content,
        "命令行参数": "argparse.ArgumentParser" in content
    }

    for name, passed in checks2.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    return all(checks.values()) and all(checks2.values())

def check_d_drive_code():
    """检查D盘代码是否存在"""
    print("\n" + "="*70)
    print("📂 检查D盘代码库")
    print("="*70)

    code_root = Path("D:/Fatigue check")
    if not code_root.exists():
        print(f"❌ D盘代码库不存在: {code_root}")
        return False

    print(f"✅ D盘代码库存在: {code_root}")

    # 检查关键文件
    key_files = [
        "fatigue_training/development.py",
        "fatigue_training/development_models.py",
        "fatigue_training/data.py",
        "fatigue_training/normalization.py"
    ]

    all_ok = True
    for rel_path in key_files:
        fpath = code_root / rel_path
        if fpath.exists():
            print(f"  ✅ {rel_path}")
        else:
            print(f"  ❌ {rel_path} 缺失!")
            all_ok = False

    return all_ok

def estimate_sizes():
    """估算文件大小"""
    print("\n" + "="*70)
    print("📊 文件大小估算（上传用）")
    print("="*70)

    deploy_dir = Path(__file__).parent
    upload_files = [
        "regression_models.py",
        "regression_pilot.py",
        "deploy_and_run.sh"
    ]

    total_size = 0
    for fname in upload_files:
        fpath = deploy_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            total_size += size
            print(f"{fname:30s} {size:>8,} bytes ({size/1024:.1f} KB)")

    print(f"{'总计':30s} {total_size:>8,} bytes ({total_size/1024:.1f} KB)")
    print(f"\n预计上传时间: <1分钟 (约{total_size/1024/1024:.2f} MB)")

def print_next_steps():
    """打印下一步操作"""
    print("\n" + "="*70)
    print("🚀 下一步操作")
    print("="*70)
    print("""
1. 确认所有检查都通过 ✅

2. 上传到服务器（在本地终端执行）:
   cd "C:/Users/dingjianbo/Documents/ChatGPT/疲劳检测/work/regression_pilot/server_deploy"
   scp regression_models.py root@YOUR_SERVER:/root/regression-pilot-code/
   scp regression_pilot.py root@YOUR_SERVER:/root/regression-pilot-code/
   scp deploy_and_run.sh root@YOUR_SERVER:/root/regression-pilot-code/

3. SSH到服务器运行:
   ssh root@YOUR_SERVER
   cd /root/regression-pilot-code
   chmod +x deploy_and_run.sh
   bash deploy_and_run.sh

4. 等待1-2小时，然后下载结果:
   scp -r root@YOUR_SERVER:/root/regression-pilot-v1 ./results/

详细指南请看: SERVER_GUIDE.md
速查卡请看: QUICK_START.md
""")

def main():
    print("🔧 回归训练Pilot - 本地预检\n")

    checks = [
        ("文件齐全", check_files()),
        ("代码结构", check_code_structure()),
        ("D盘代码", check_d_drive_code())
    ]

    estimate_sizes()

    # 总结
    print("\n" + "="*70)
    print("📝 预检总结")
    print("="*70)

    all_passed = all(result for _, result in checks)

    for name, result in checks:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s} {status}")

    if all_passed:
        print("\n🎉 所有检查通过！可以上传到服务器了。")
        print_next_steps()
        return 0
    else:
        print("\n⚠️ 有检查未通过，请先解决问题。")
        return 1

if __name__ == '__main__':
    sys.exit(main())
