from __future__ import annotations
import hashlib
import json
import random
import shutil
from pathlib import Path
from .common import atomic_json, digest, file_hash

DATA_HASHES = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}
AUTHOR_BLOBS = {
    "gsm/gen_gsm.py": "affc073fe2d6718a5c0ca3c02921bc82616671fe",
    "gsm/eval_gsm.py": "7715569d55120e0d9da39df23b7f251ca992b72d",
    "requirements.txt": "2d176347ff0684f996e30e8b479fc0c7d6b90914",
    "README.md": "72804b8ac3621299022783e99a8384f96db52c56",
}
TEST_QUESTIONS_HASH = "df7ad37de362042700e1aa26efe93da2bd39fd22647f75c6e730e04e246515ea"


def read_data(directory, split):
    path = Path(directory) / (split + ".jsonl")
    if file_hash(path) != DATA_HASHES[split]:
        raise ValueError("Dataset hash differs from approved GSM8K version: " + split)
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        item = json.loads(line)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError("Invalid question")
        if not isinstance(item.get("answer"), str) or "####" not in item["answer"]:
            raise ValueError("Invalid gold")
        records.append(dict(source_line=line_no, question=item["question"], answer=item["answer"]))
    return records


def manifest(records, split, purpose):
    prefix = "" if split == "test" else purpose + "_"
    rows = [dict(qid=prefix + f"{i:04d}", split=split, **item,
                 question_sha256=hashlib.sha256(item["question"].encode()).hexdigest())
            for i, item in enumerate(records, 1)]
    doc = dict(schema_version=1, split=split, purpose=purpose, dataset_sha256=DATA_HASHES[split],
               sampling=dict(algorithm="random.Random(0).shuffle", seed=0),
               questions_sha256=hashlib.sha256("\n".join(r["question"] for r in rows).encode()).hexdigest(),
               records=rows)
    return dict(doc, manifest_sha256=digest(doc))


def load_manifest(path):
    from .scoring import parse_gold
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    original = {k: v for k, v in value.items() if k != "manifest_sha256"}
    if value.get("manifest_sha256") != digest(original):
        raise ValueError("Manifest fingerprint mismatch")
    rows = value["records"]
    if not rows or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Empty manifest or duplicate qid")
    for row in rows:
        if row["split"] != value["split"] or hashlib.sha256(row["question"].encode()).hexdigest() != row["question_sha256"]:
            raise ValueError("Record split or question fingerprint mismatch")
        parse_gold(row["answer"])
    return value


def prepare(data_directory, author_directory, project):
    project, author_directory = Path(project), Path(author_directory)
    train, test = read_data(data_directory, "train"), read_data(data_directory, "test")
    random.Random(0).shuffle(train)
    random.Random(0).shuffle(test)
    docs = {
        "train_debug_10.json": manifest(train[:10], "train", "debug"),
        "train_calibration_30.json": manifest(train[10:40], "train", "calibration"),
        "questions_100.json": manifest(test[:100], "test", "evaluation"),
    }
    if docs["questions_100.json"]["questions_sha256"] != TEST_QUESTIONS_HASH:
        raise ValueError("Test question order differs from approved conditions")
    sets = [set(r["question"] for r in doc["records"]) for doc in docs.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Question overlap between manifests")
    from .scoring import parse_gold
    for doc in docs.values():
        for row in doc["records"]:
            parse_gold(row["answer"])
    sources = {}
    for name, expected in AUTHOR_BLOBS.items():
        path = author_directory / name
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        blob = hashlib.sha1(b"blob " + str(len(normalized)).encode() + b"\0" + normalized).hexdigest()
        if blob != expected:
            raise ValueError("Author source differs: " + name)
        sources[name] = dict(git_blob_lf=blob, sha256=file_hash(path))
    outputs = {project / "manifests" / name: doc for name, doc in docs.items()}
    provenance = dict(author_commit="9846749350eb917ae5bfaaff4c645fc705b8d3af",
                      author_files=sources, datasets=DATA_HASHES)
    outputs[project / "author_reference" / "provenance.json"] = provenance
    # Validate every existing target before making any change.
    for path, doc in outputs.items():
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != doc:
            raise ValueError("Refusing to overwrite a different prepared artifact: " + str(path))
    for name in ("gen_gsm.py", "eval_gsm.py"):
        target = project / "author_reference" / name
        source = author_directory / "gsm" / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("Reference copy differs")
    for path, doc in outputs.items():
        if not path.exists():
            atomic_json(path, doc)
    for name in ("gen_gsm.py", "eval_gsm.py"):
        target = project / "author_reference" / name
        if not target.exists():
            shutil.copyfile(author_directory / "gsm" / name, target)
    return {name: doc["manifest_sha256"] for name, doc in docs.items()}
