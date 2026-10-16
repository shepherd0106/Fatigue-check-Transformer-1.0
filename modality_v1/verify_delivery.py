"""Verify this standalone diagnostic delivery before any server mutation."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def linked(part):
    try:attributes=getattr(part.lstat(),'st_file_attributes',0)
    except FileNotFoundError:attributes=0
    return part.is_symlink() or bool(attributes&0x400)


def verify(root):
    root=Path(root).absolute()
    for part in (root,*root.parents):
        if linked(part):raise ValueError('Linked delivery directory')
    root=root.resolve()
    index=json.loads((root/'DELIVERY_INDEX.json').read_text(encoding='utf-8'))
    for name,want in index['files'].items():
        item=PurePosixPath(name)
        if item.is_absolute() or '..' in item.parts or item.as_posix()!=name or ':' in name or '\\' in name:
            raise ValueError('Unsafe delivery name')
        path=root.joinpath(*item.parts)
        for part in (path,*path.parents):
            if linked(part):raise ValueError('Linked delivery member: '+name)
        if path.is_symlink() or not path.is_file():raise ValueError('Missing/linked delivery file: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise ValueError('Delivery changed: '+name)
    return len(index['files'])


if __name__=='__main__':
    print('Delivery verified:',verify(Path(__file__).parent),'files; no training executed')
