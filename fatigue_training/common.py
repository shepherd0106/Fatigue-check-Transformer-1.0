"""JSON provenance and atomic output helpers (no torch dependency)."""
# 公共工具：读写 JSON、计算 SHA256 指纹、保存失败尝试的文件。
# 这些函数帮助追踪结果对应的数据和配置，不参与神经网络前向或梯度计算。
import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(value):
    # 为可序列化的结构生成指纹；字典键排序，使键的插入顺序不影响结果。
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    # 文件按1MB分块读取并计算内容哈希，避免一次读取大文件占用过多内存。
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8 文件。
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    # 先完整写入临时文件，再替换目标；其他读取者不会看到写到一半的 JSON。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def archive_incomplete_attempt(output):
    """Preserve failed/interrupted evidence before restarting a fit from epoch one."""
    # 将未完成 fit 的现有文件复制到 attempts/1、attempts/2…，再允许重跑。
    # completed 的结果应通过 training.reusable 核对后复用，不能在这里覆盖。
    output = Path(output)
    if not (output/"status.json").exists():
        return None
    if read_json(output/"status.json").get("state") == "completed":
        raise ValueError("Completed attempts must be verified/reused, never overwritten")
    attempts = output/"attempts"
    attempts.mkdir(exist_ok=True)
    number = max([int(p.name) for p in attempts.iterdir() if p.is_dir() and p.name.isdigit()]+[0])+1
    dest = attempts/str(number)
    dest.mkdir()
    for source in output.iterdir():
        if source.is_file():
            shutil.copy2(source, dest/source.name)
    return dest
