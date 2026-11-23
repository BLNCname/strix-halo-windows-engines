"""Package the native executable and the license notices of its pinned source."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

VERSION = "1.0.0"
PROFILES = ["Q4_K_XL", "IQ4_XS", "Q3_K_XL"]


def package(exe, source, output):
    files = {"gufo.exe": exe, **{name: source / name for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")}}
    files.update({"third-party/" + p.name: p for p in sorted((source / "licenses").glob("*.txt"))})
    files["third-party/ds4.txt"] = source / "src/models/deepseek_v4_flash/LICENSE.ds4"
    if "third-party/llama.cpp.txt" not in files or "third-party/hip.txt" not in files:
        raise ValueError("Gufo source notices are missing")
    data = {name: path.read_bytes() for name, path in files.items()}
    manifest = {"schema": 1, "version": VERSION, "native": "0.35.0-strix.1", "target": "windows-x64-gfx1151",
                "context": 262144, "profiles": PROFILES,
                "files": {name: hashlib.sha256(buf).hexdigest() for name, buf in data.items()}}
    data["engine.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x") as archive:
        for name, buf in sorted(data.items()):
            entry = zipfile.ZipInfo(name)
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, buf, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    print(hashlib.sha256(output.read_bytes()).hexdigest(), output.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "source", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    package(args.exe, args.source, args.output)
