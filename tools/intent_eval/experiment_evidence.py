import hashlib
import json
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_evidence(path):
    root = Path(path)
    return {"path": str(root.resolve()), "files": {
        name: file_hash(root / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")}}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    import time
    for attempt in range(6):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (attempt + 1))


def require_matching_evidence(actual, expected):
    if actual != expected:
        raise ValueError("实验输入或模型指纹变化，禁止混用断点结果")


def training_provenance(adapter, train, allow_unverified=False):
    path = Path(adapter) / "training_manifest.json"
    if not path.exists():
        if not allow_unverified:
            raise ValueError("模型没有训练来源 manifest；历史模型只能显式允许未验证来源，不得用于留出验收")
        return {"status": "unverified-legacy", "declared_train_sha256": file_hash(train)}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["train_sha256"] != file_hash(train) or manifest["adapter"] != adapter_evidence(adapter):
        raise ValueError("adapter 与训练输入 manifest 不匹配")
    if manifest.get("parent_adapter"):
        return {"status": "verified-current-stage-unverified-ancestry",
                "manifest_sha256": file_hash(path),
                "parent_adapter": manifest["parent_adapter"]}
    return {"status": "verified", "manifest_sha256": file_hash(path)}
