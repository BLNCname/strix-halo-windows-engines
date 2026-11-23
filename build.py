"""Build the Flash-Next edition of Gufo in an empty work directory on Windows."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
UPSTREAM = "https://github.com/gufo-org/gufo.git"
REVISION = "0260e1162f8713caee4d3f4ff77e017de7165f9c"
VERSION = "0.35.0-strix.2"


def build(work, cmake_args, jobs):
    work = work.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError("Choose an empty work directory")
    work.mkdir(parents=True, exist_ok=True)
    src = work / "gufo"
    out = work / "build"
    subprocess.run(["git", "clone", "--depth=1", "--branch=v0.35.0", UPSTREAM, src], check=True)
    head = subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        raise ValueError(f"Unexpected upstream revision {head}")
    for patch in ("mixed-gguf.patch", "kv-cache.patch"):
        subprocess.run(["git", "-C", src, "apply", "--whitespace=error", ROOT / patch], check=True)
    env = os.environ.copy()
    # Rust panic locations and clang __FILE__ can otherwise retain profile paths.
    roots = [(work, "build"), (Path.home(), "profile")]
    for key in ("HIP_PATH", "CARGO_HOME", "RUSTUP_HOME"):
        if env.get(key):
            roots.insert(0, (Path(env[key]).resolve(), key.lower()))
    mappings = [(p, name) for root, name in roots for p in dict.fromkeys([str(root), root.as_posix()])]
    clang_maps = " ".join(f'"/clang:-ffile-prefix-map={p}={name}"' for p, name in mappings)
    rust_flags = env.get("CARGO_ENCODED_RUSTFLAGS", "").split("\x1f") if env.get("CARGO_ENCODED_RUSTFLAGS") else []
    rust_flags.extend(f"--remap-path-prefix={p}={name}" for p, name in reversed(mappings))
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(rust_flags)
    for key in ("CFLAGS", "CXXFLAGS", "HIPFLAGS"):
        env[key] = (env.get(key, "") + " " + clang_maps).strip()
    # The default preset uses vcpkg and the Visual Studio developer environment.
    configure = cmake_args or ["--preset", "windows-release"]
    subprocess.run(["cmake", "-S", src, "-B", out, *configure, "-DCMAKE_BUILD_TYPE=Release",
                    "-DGUFO_EXTENSIONS=qwen38_flash_next", "-DGUFO_BUILD_TOOLS=OFF", "-DBUILD_TESTING=OFF",
                    "-DCMAKE_HIP_FLAGS_RELEASE=/clang:-O3 /DNDEBUG", "-DCMAKE_HIP_ARCHITECTURES=gfx1151",
                    f"-DGUFO_RELEASE_VERSION={VERSION}", "-DGUFO_REVISION=0260e11-strix.2"], cwd=src, env=env, check=True)
    subprocess.run(["cmake", "--build", out, "--config", "Release", "--target", "gufo", "--parallel", str(jobs)],
                   cwd=src, env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--jobs", type=int, default=12)
    options = sys.argv[1:]
    split = options.index("--") if "--" in options else len(options)
    args = parser.parse_args(options[:split])
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args.work, options[split + 1:], args.jobs)
