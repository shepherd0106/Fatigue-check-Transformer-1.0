"""Manifest/CSV indexed mmap windows. Invalid source windows never reappear."""
# 模块职责：把磁盘上的连续信号和窗口索引转成训练可用的 NumPy 批次。
# manifest.json 描述文件与采集 run（一次连续记录）；windows.csv 描述窗口。
# 本模块读取既有0/1标签，不从原始信号计算疲劳标签，也不执行采集预处理。
# 常用调用：WindowDataset(...) → indices(被试名单) → batch(窗口索引,统计量)。
import csv
import hashlib
import io
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from .common import digest, file_hash, read_json, write_json

MODALITIES = {"eeg": (4, 2500), "fnirs": (24, 250)}
# 每种模态的 (信号行数,10秒窗口采样点数)；行数与 CNN 中的特征图数不同。
CSV_FIELDS = ["source_index", "subject_id", "run_id", "block_info", "window_in_run",
              "eeg_start", "eeg_end", "fnirs_start", "fnirs_end", "label_2class",
              "paired_valid", "outer_test_fold", "invalid_reasons"]
STAGES = [f"block{i}" for i in range(1, 13)] + [f"rest{i}" for i in range(1, 5)]


@dataclass(frozen=True)
class Window:
    # 一个窗口的元数据，不含信号数组；frozen=True 防止加载后随意改写字段。
    # source_index 是原始窗口唯一编号；窗口在 self.windows 中的位置是另一种索引。
    # *_start/end 为连续数组上的半开区间 [start,end)，end 对应点不包含在内。
    source_index: int
    subject_id: str
    run_id: str
    block_info: str
    window_in_run: int
    eeg_start: int
    eeg_end: int
    fnirs_start: int
    fnirs_end: int
    label_2class: int
    paired_valid: bool
    outer_test_fold: int
    invalid_reasons: str


def merge_intervals(intervals):
    # 合并重叠或相接的区间，供标准化统计使用，避免重叠窗口重复计算采样点。
    # 例：[(0,2500),(1250,3750)] → [[0,3750]]。
    merged = []
    for start, end in sorted(intervals):
        if start < 0 or end <= start:
            raise ValueError("Invalid continuous interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


class WindowDataset:
    # 这是项目自定义的数据容器，不是 torch.utils.data.Dataset 子类。
    # 信号按需加载并缓存；没有使用 PyTorch DataLoader，批次由 batch 显式组装。
    def __init__(self, manifest, data_root=None):
        # 第1步：读取清单，确定物理文件根目录，并记录文件状态以检测后续变化。
        self.manifest_path = Path(manifest).resolve()
        self.manifest = read_json(self.manifest_path)
        self._snapshots = {self.manifest_path: self._stat(self.manifest_path)}
        self._manifest_hash = file_hash(self.manifest_path)
        m = self.manifest
        if m.get("schema_version") != 1 or m.get("kind") not in ("synthetic", "real"):
            raise ValueError("Manifest schema_version=1 and kind=synthetic/real required")
        if not isinstance(m.get("data_version"), str) or not m["data_version"] or not m.get("source"):
            raise ValueError("Data version and source provenance required")
        self.root = Path(data_root).resolve() if data_root else (self.manifest_path.parent / m.get("data_root", ".")).resolve()
        self.index_path = (self.manifest_path.parent / m["window_index"]).resolve()
        self.runs = {}
        array_owners = {}
        # 第2步：建立 run_id→元数据 的查找表，检查每个模态的文件描述。
        for run in m["runs"]:
            key = str(run["run_id"])
            if key in self.runs or not str(run.get("subject_id", "")) or run.get("block_info") not in STAGES:
                raise ValueError("Duplicate/invalid run metadata")
            for modality in MODALITIES:
                files = run[modality]
                if not files.get("array") or not (files.get("coverage") or files.get("all_samples_covered") is True):
                    raise ValueError("Each modality requires array and coverage or explicit all_samples_covered")
                resolved = (self.root / files["array"]).resolve()
                if resolved in array_owners:
                    # 防止不同 run/被试误指向同一信号文件，污染实验划分。
                    raise ValueError("Distinct runs must not reuse the same continuous signal file")
                array_owners[resolved] = key
            self.runs[key] = run
        self.windows = []
        seen = set()
        positions = set()
        self._snapshots[self.index_path] = self._stat(self.index_path)
        csv_bytes = self.index_path.read_bytes()
        # 第3步：解析所有原始窗口（包括无效窗口），保留原始编号与无效原因。
        # 使用同一份 CSV 字节解析和计算哈希，保证指纹对应实际读取内容。
        self._csv_hash = hashlib.sha256(csv_bytes).hexdigest()
        with io.StringIO(csv_bytes.decode("utf-8-sig"), newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(CSV_FIELDS).issubset(reader.fieldnames or []):
                raise ValueError("Window CSV missing required fields")
            for raw in reader:
                if raw["paired_valid"] not in ("0", "1"):
                    raise ValueError("paired_valid must be exactly 0 or 1")
                values = {k: int(raw[k]) for k in CSV_FIELDS if k not in
                          ("subject_id", "run_id", "block_info", "invalid_reasons", "paired_valid")}
                w = Window(**values, subject_id=str(raw["subject_id"]), run_id=str(raw["run_id"]),
                           block_info=raw["block_info"], invalid_reasons=raw["invalid_reasons"],
                           paired_valid=raw["paired_valid"] == "1")
                # paired_valid=True 表示该原窗口在两种模态上共同有效。
                if w.source_index < 0 or w.source_index in seen or (w.run_id, w.window_in_run) in positions:
                    raise ValueError("Duplicate/negative original source index or window position")
                seen.add(w.source_index)
                positions.add((w.run_id, w.window_in_run))
                if w.window_in_run < 0 or w.label_2class not in (0, 1) or w.outer_test_fold not in range(1, 6):
                    raise ValueError("Invalid label/window position/outer fold")
                run = self.runs.get(w.run_id)
                if run is None or str(run["subject_id"]) != w.subject_id or run["block_info"] != w.block_info:
                    raise ValueError("Window/run metadata mismatch")
                # The CSV explicitly defines eligibility; array reconstruction cannot override it.
                # 是否可用以原始 CSV 为准；完整 coverage 不能把无效窗自动变成有效窗。
                for modality, (_, length) in MODALITIES.items():
                    start, end = getattr(w, modality+"_start"), getattr(w, modality+"_end")
                    if start < 0 or end - start != length:
                        raise ValueError("Window intervals must have the fixed modality length")
                if w.paired_valid and w.invalid_reasons:
                    raise ValueError("Valid window has exclusion reasons")
                self.windows.append(w)
        if not self.windows or not any(w.paired_valid for w in self.windows):
            raise ValueError("No common valid windows")
        self._arrays = {}
        # 构造函数只解析元数据；真正访问某个 run 时再用 mmap 读取信号。
        self._fingerprint = None
        self.check_unchanged()

    @staticmethod
    def _stat(path):
        # 记录文件大小和修改/状态时间，用于识别加载之后的常见数据变化。
        s = Path(path).stat()
        return (s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def check_unchanged(self):
        # 若文件在训练期间发生改变，立即报错，防止统计量与训练数据不一致。
        for path, snapshot in self._snapshots.items():
            if self._stat(path) != snapshot:
                raise ValueError("Data changed after loading; create a new Dataset instance")

    def indices(self, subjects, nonoverlap=False):
        # 返回 self.windows 的位置索引，仅选择名单中的共同有效窗口。
        # 非重叠诊断使用原 window_in_run 的偶数编号；不对筛选后的窗口重新编号。
        # 在原始5秒步长下，每隔一个10秒窗口取一次，形成非重叠诊断子集。
        subjects = set(map(str, subjects))
        return [i for i, w in enumerate(self.windows) if w.subject_id in subjects and w.paired_valid
                and (not nonoverlap or w.window_in_run % 2 == 0)]

    def check_cohort(self, cfg):
        # 检查数据是否完整覆盖协议的30人，CSV 折编号是否与固定分组一致。
        expected = {str(s): i for i, g in enumerate(cfg["outer_groups"], 1) for s in g}
        if {w.subject_id for w in self.windows} != set(expected):
            raise ValueError("Dataset cohort differs from protocol; revise protocol before formal runs")
        if any(w.outer_test_fold != expected[w.subject_id] for w in self.windows):
            raise ValueError("CSV outer fold differs from fixed subject grouping")
        if any(not self.indices([s]) for s in expected):
            raise ValueError("Protocol subject has no common valid windows")

    def arrays(self, run_id, modality):
        # 同一 run/模态只加载一次。mmap_mode='r' 建立只读磁盘映射，
        # 无需一次把整段信号复制进内存；allow_pickle=False 禁止加载对象数组。
        key = (str(run_id), modality)
        if key not in self._arrays:
            files = self.runs[key[0]][modality]
            arr = np.load(self.root / files["array"], mmap_mode="r", allow_pickle=False)
            rows, _ = MODALITIES[modality]
            if arr.ndim != 2 or arr.shape[0] != rows or arr.dtype != np.dtype("float32"):
                raise ValueError(f"Invalid continuous {modality} array shape/dtype")
            mask = None
            if files.get("coverage"):
                # coverage 每个采样点一个布尔值，表示该点是否有覆盖。
                mask = np.load(self.root / files["coverage"], mmap_mode="r", allow_pickle=False)
                if mask.shape != (arr.shape[1],) or mask.dtype != np.dtype("bool"):
                    raise ValueError("Coverage mask must be bool [continuous_samples]")
            self._arrays[key] = arr, mask
        return self._arrays[key]

    def slice(self, w, modality, start=None, end=None):
        # 提取窗口信号；标准化合并区间时也可传入自定义 start/end。
        # 检查边界、coverage 和有限值；返回独立副本，避免改写只读映射。
        arr, mask = self.arrays(w.run_id, modality)
        start = getattr(w, modality+"_start") if start is None else start
        end = getattr(w, modality+"_end") if end is None else end
        if end > arr.shape[1] or mask is not None and not np.all(mask[start:end]):
            raise ValueError("Eligible interval exceeds continuous array or contains uncovered samples")
        out = np.array(arr[:, start:end], dtype=np.float32, copy=True)
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite signal in eligible coverage")
        return out

    def batch(self, indices, stats):
        # 输入：一批窗口位置索引，以及只用训练被试拟合的标准化统计量。
        # 输出：eeg [B,4,2500]、fnirs [B,24,250]（float32）、标签 [B]（int64）。
        windows = [self.windows[int(i)] for i in indices]
        if not windows or any(not w.paired_valid for w in windows):
            raise ValueError("Batch must contain only common valid windows")
        modalities = []
        for modality in MODALITIES:
            s = stats["modalities"][modality]
            mean = np.asarray(s["mean"], dtype=np.float64)[:, None]
            # [C] → [C,1]，利用广播为同一通道的所有时间点减均值、除标准差。
            std = np.asarray(s["std"], dtype=np.float64)[:, None]
            x = np.stack([((self.slice(w, modality).astype(np.float64)-mean)/std).astype(np.float32) for w in windows])
            # 每窗做 z-score：(x-mean)/std；np.stack 在最前面增加批次轴。
            # 标准化运算使用 float64，之后转回模型要求的 float32。
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite normalized batch")
            modalities.append(x)
        return (*modalities, np.array([w.label_2class for w in windows], dtype=np.int64))

    def fingerprint(self):
        """Strong content fingerprint; explicit validation/training scans all referenced NPY bytes."""
        # 指纹由清单、窗口 CSV 和所有引用的 NPY 文件内容共同确定。
        # 第一次会读取文件字节做哈希；以后缓存指纹，但仍检查文件是否变化。
        # 这与“仅用训练被试拟合统计量”是两件事：哈希不参与学习模型。
        self.check_unchanged()
        if self._fingerprint is None:
            payload = dict(self.manifest)
            payload.pop("data_root", None)  # Physical relocation leaves scientific content unchanged.
            files = {}
            for run in self.runs.values():
                for modality in MODALITIES:
                    for key in ("array", "coverage"):
                        path = run[modality].get(key)
                        if path:
                            resolved = (self.root / path).resolve()
                            self._snapshots[resolved] = self._stat(resolved)
                            files[path] = file_hash(resolved)
            self.check_unchanged()
            self._fingerprint = digest(dict(manifest=payload, csv=self._csv_hash, files=files))
        return self._fingerprint

    def acceptance(self, receipt):
        # 正式实验要求真实数据的验收凭据：时间轴、fNIRS转换、标签均已验收，
        # 并与当前数据指纹/版本相符；代码核对凭据字段，不代替实际科研验收。
        if self.manifest["kind"] == "synthetic":
            raise ValueError("Synthetic data cannot be marked formal real-data evidence")
        a = read_json(receipt)
        if (a.get("data_fingerprint") != self.fingerprint() or a.get("data_version") != self.manifest["data_version"]
                or not a.get("approved_by") or not a.get("approved_at")
                or any(a.get(k) != "passed" for k in ("time_axis", "fnirs_conversion", "labels"))):
            raise ValueError("Acceptance receipt missing, incomplete, or bound to other data")
        return dict(receipt_sha256=file_hash(receipt), receipt=a)


def adapt_legacy(manifest_path, window_csv, output, data_root):
    """Metadata-only adapter: no PKL/XDF conversion, no NPY read, no acceptance inferred."""
    # 将旧文件目录描述映射为新 manifest；保留原 CSV 字节，仅改数据接口描述。
    legacy = read_json(manifest_path)
    runs = []
    for run in legacy["runs"]:
        directory = str(run["relative_directory"]).replace("\\", "/")
        runs.append(dict(run_id=str(run["run_id"]), subject_id=str(run["subject_id"]), block_info=run["block_info"],
                         **{m: dict(array=f"{directory}/data_{m}.npy", coverage=f"{directory}/data_{m}_coverage.npy")
                            for m in MODALITIES}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Exact CSV bytes preserve invalidity, source indices, and original parity.
    (output / "windows.csv").write_bytes(Path(window_csv).read_bytes())
    m = dict(schema_version=1, kind="real", data_version="legacy-adapted-"+file_hash(window_csv)[:16],
             data_root=str(Path(data_root).resolve()), window_index="windows.csv",
             source=dict(legacy_manifest_sha256=file_hash(manifest_path), original_time_axis="nominal_unverified",
                         preprocessing="legacy_unverified", label_definition="stored_F_mean_ge_0.4",
                         source_pickle_basename=Path(str(legacy.get("source_pickle", "")).replace("\\", "/")).name), runs=runs)
    write_json(output / "manifest.json", m)
    return output / "manifest.json"
