"""Small deterministic fixtures, never accepted as real research evidence."""
# 模块职责：生成与真实数据接口相同的小型合成数据，方便检查读取与训练链路。
# 合成信号来自随机数，标签来自人为奇偶规则，不能解释为真实疲劳识别效果。
import csv
from pathlib import Path
import numpy as np
from .common import write_json
from .data import CSV_FIELDS, MODALITIES
from .protocol import load_config


def generate(output, windows_per_subject=3, seed=20261011):
    # 默认每人3个重叠窗口，覆盖协议的30人；固定 seed 使生成结果可重复。
    if windows_per_subject < 2:
        raise ValueError("At least two windows per subject required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"manifest.json").exists():
        raise FileExistsError("Synthetic destination already contains a manifest")
    cfg = load_config()
    rng = np.random.default_rng(seed)
    runs, windows = [], []
    source = 0
    for outer, subjects in enumerate(cfg["outer_groups"], 1):
        # 每人生成一个 block1 run，并保存独立的 EEG/fNIRS 数组及 coverage。
        for subject in subjects:
            sid = str(subject)
            directory = output/f"subject_{sid}"
            directory.mkdir(exist_ok=True)
            run = dict(run_id=sid, subject_id=sid, block_info="block1")
            for modality, (rows, length) in MODALITIES.items():
                count = length+(windows_per_subject-1)*(length//2)
                # 连续长度=首窗长度+后续窗口的半窗步长，确保所有切片都在数组内。
                arr = rng.normal(size=(rows, count)).astype(np.float32)
                # Deliberately synthetic signal; this score has no fatigue interpretation.
                arr += np.float32(subject*.01)
                np.save(directory/f"{modality}.npy", arr)
                np.save(directory/f"{modality}_coverage.npy", np.ones(count, dtype=bool))
                run[modality] = dict(array=f"subject_{sid}/{modality}.npy", coverage=f"subject_{sid}/{modality}_coverage.npy")
            runs.append(run)
            for j in range(windows_per_subject):
                # One invalid original position demonstrates preservation despite full reconstruction coverage.
                # 特意保留一个无效原窗：即使信号覆盖完整，数据容器仍须按 CSV 排除。
                valid = not (sid == "25" and j == 1)
                windows.append(dict(source_index=source, subject_id=sid, run_id=sid, block_info="block1",
                                    window_in_run=j, eeg_start=1250*j, eeg_end=1250*j+2500,
                                    fnirs_start=125*j, fnirs_end=125*j+250, label_2class=(subject+j)%2,
                                    paired_valid=int(valid), outer_test_fold=outer,
                                    invalid_reasons="synthetic_original_invalid" if not valid else ""))
                source += 1
    with (output/"windows.csv").open("w", newline="", encoding="utf-8") as stream:
        # CSV 保存窗口索引，manifest 保存文件映射，格式与 data.WindowDataset 对应。
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(windows)
    write_json(output/"manifest.json", dict(schema_version=1, kind="synthetic", data_version=f"synthetic-{seed}-{windows_per_subject}",
                 data_root=".", window_index="windows.csv", source=dict(generator="fatigue_training.synthetic", seed=seed,
                 label_definition="synthetic_parity_no_physiological_meaning"), runs=runs))
    return output/"manifest.json"
