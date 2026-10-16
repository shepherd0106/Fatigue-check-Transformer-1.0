"""可选监控：不改变模型、随机源、抽样或科学配置。"""
import json
import math
from pathlib import Path


def writer_factory():
    try:
        import tensorboard  # noqa: F401; only opt-in calls require this dependency
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError("TensorBoard monitoring requires torch and tensorboard; install the optional monitoring extra on the server") from error
    return SummaryWriter


def check_logging(options=None):
    options = options or {}
    if set(options)-{"tensorboard"}:
        raise ValueError("Unknown logging option")
    if options.get("tensorboard"):
        writer_factory()  # Fail before new training state or data scanning.


class EpochLogger:
    """每次尝试单独event目录；即使重跑fit也不拼接旧尝试的曲线。"""
    def __init__(self, output, factory=None):
        root = Path(output)/"tensorboard"
        root.mkdir(parents=True, exist_ok=True)
        number = max([int(p.name.split("_")[-1]) for p in root.glob("attempt_*")
                      if p.is_dir() and p.name.split("_")[-1].isdigit()]+[0])+1
        self.directory = root/f"attempt_{number}"
        self.directory.mkdir()
        self.writer = (factory or writer_factory())(log_dir=str(self.directory))
        self.closed = False

    def write(self, record, learning_rate=None, is_refit=False):
        if self.closed:
            raise RuntimeError("TensorBoard writer already closed")
        values = {"train/loss": record.get("mean_loss"), "time/elapsed_seconds": record.get("seconds"),
                  "optimizer/learning_rate": learning_rate,
                  "resource/peak_cuda_memory_bytes": record.get("peak_cuda_memory_bytes")}
        if not is_refit:
            values["inner_validation/primary"] = record.get("primary_validation")
        try:
            epoch = int(record["epoch"])
            if epoch < 1:
                raise ValueError("Epoch must be positive")
            for tag, value in values.items():
                if value is not None:
                    if not math.isfinite(float(value)):
                        raise ValueError(f"Nonfinite TensorBoard scalar {tag}")
                    self.writer.add_scalar(tag, float(value), epoch)
            self.writer.flush()
        except Exception as error:
            raise RuntimeError(f"TensorBoard log write failed in {self.directory}: {error}") from error

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.writer.flush()
            finally:
                self.writer.close()
        except Exception as error:
            raise RuntimeError(f"TensorBoard flush/close failed in {self.directory}: {error}") from error


def export_logs(results, output=None, factory=None):
    """将既有JSONL转为events；只读文本，不加载模型或信号。"""
    factory = factory or writer_factory()
    root = Path(results).resolve()
    destination = Path(output).resolve() if output else root/"tensorboard_export"
    exported = []
    for path in sorted(root.glob("*/outer_*/*/epochs.jsonl")):
        configuration = path.parent/"configuration.json"
        if not configuration.exists():
            continue
        from .common import read_json
        task = read_json(configuration)["task"]
        logger = EpochLogger(destination/path.parent.relative_to(root), factory)
        try:
            previous = 0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if int(record["epoch"]) <= previous:
                        raise ValueError(f"Non-increasing epoch sequence: {path}")
                    previous = int(record["epoch"])
                    logger.write(record, task.get("lr"), task["stage"] == "refit")
            exported.append(dict(source=str(path), events=str(logger.directory), epochs=previous))
        finally:
            logger.close()
    return dict(exported=exported, count=len(exported), output=str(destination), signals_or_checkpoints_loaded=False)
