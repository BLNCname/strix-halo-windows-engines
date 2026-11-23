huge credit goes to [@wenis](https://github.com/wenis), [Peonist](https://github.com/peonist-ai) and the [HaloGen Flash Server project](https://github.com/peonist-ai/halogen-flash-server) for their work on Flash-Next inference on Strix Halo.

# Strix Halo Windows engines

Three selectable ROCm profiles run the standard Unsloth Qwen3.8-Flash-Next GGUF shards on Windows 11 and Radeon 8060S. They share a native executable built from [Gufo v0.35.0](https://github.com/gufo-org/gufo/tree/v0.35.0). Gufo implements the model, loader and server. Its quantized matrix kernels incorporate code from [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp). HaloGen is an inspiration for this project. The executable builds from Gufo and the patch in this repository.

| Profile | Routed expert matrices in the four Unsloth shards |
| --- | --- |
| Q4_K_XL | 94 Q4_K, 43 Q5_1, 5 Q8_0, 2 Q5_K |
| IQ4_XS | 56 IQ3_XXS, 34 IQ4_XS, 33 Q5_0, 13 IQ4_NL, 6 IQ2_S, 2 Q2_0 |
| Q3_K_XL | 76 IQ3_XXS, 26 IQ4_NL, 20 Q5_0, 12 IQ4_XS, 8 IQ2_S, 2 Q2_0 |

The loader dispatches by tensor type. Both mixed profiles retain 142 packed expert matrices and expand two Q2_0 matrices to Q8_0 exactly on upload. Dense Q6_K and the IQ4_XS token embedding expand to FP16. The 26.82 GiB IQ4_NL per-layer embedding uses CPU row lookup. The default reads requested rows from its shard and keeps a bounded RAM cache. An explicit lazy-mode off loads the complete packed table into host memory. The native context is 262144 tokens. The optional external Q8_0 MTP head is a separate GGUF. Obtain weights from [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

### Attention cache in v1.1.0

The target and MTP head store K and V in real GGML Q8_0, Q4_0 or FP16 rows. For direct Gufo invocation set `GUFO_FLASH_CACHE_K` and `GUFO_FLASH_CACHE_V` to `q8_0`, `q4_0` or `f16`. The draft inherits both values. Optional overrides are `GUFO_FLASH_DRAFT_CACHE_K` and `GUFO_FLASH_DRAFT_CACHE_V`. A direct invocation with no format settings retains FP16. The loader prints all four resolved formats.

Each group of 32 values uses 34 bytes in Q8_0 or 18 bytes in Q4_0. The FP16 representation takes 64 bytes. Narrow attention reads packed rows directly. Wide prefill uses one FP16 work buffer per quantized side shared by all layers and sessions. Both saved continuation data and admission checks account for the selected row layout. Recurrent state and the sparse index use their model-specific types.

On a 128 GiB Radeon 8060S system, a Q3 load at context 262144 with four sessions and a separate MTP head peaked at 122.33 GiB system commit with FP16 KV and 110.57 GiB with Q8_0 KV. Windows pagefile occupancy was about 1.3 GiB in both runs. For Q4 with the same window and four sessions, FP16 admission required four 7.01 GiB states with only 24.7 GiB available. Q8_0 loaded and served a real Hermes tool request.

### Launch arguments and shared context

The open native entry point is `gufo serve lmstudio WORKER_PORT ALIAS -- MODEL_OPTIONS...`. A compatible host passes its complete model-launch argument vector, including one-time overrides, in the original order. A host bridge owns its transport credentials. The standalone server binds to loopback. The native process reports resolved settings and option names under `/props → studio_load`. Engine-native options continue through Gufo's own CLI parser. Unknown options and incompatible values produce a load error.

| Host controls | Native behavior |
| --- | --- |
| Context, parallel sessions, unified KV and per-slot cap | A shared token budget with HIP virtual memory, or an independent context per slot. A conversation can use the entire shared pool while other slots are idle. Each session owns its token history, recurrent state and mapped rows. |
| K/V precision, draft model and sampling gates | FP16, Q8_0 or Q4_0 independently for K and V. An external MTP head inherits the formats unless draft overrides are supplied. Its minimum and maximum proposal lengths and continuation probability are applied. |
| Flash Attention, logical batch, physical microbatch | The selected attention route and scheduler/device limits are applied. Smaller microbatches change prefill throughput. |
| Threads, loading mode, lazy rows and checkpoints | CPU worker limits, mapped/buffered/direct reads, OS locking on request, resident or on-demand PLE rows, and checkpoint retention. The OS can refuse a requested lock. |
| Template, projector, reasoning and RoPE | Validates the selected template against the GGUF Qwen template. Loads the selected BF16 or F16 projector. Applies supported native prediction defaults and frequency overrides. |
| GPU placement | The Strix Halo path requires one GPU, complete trunk/expert offload and device K/V. An incompatible placement, custom Jinja program or unsupported cache type is reported by name. |

A live launch emitted 32 options including transport authentication. The native worker received all 58 remaining argument strings (31 option names) with identical order and values. A real SDK override produced **four separate 8192-token windows**, Flash Attention off, batch 384, microbatch 128, zero checkpoints, Q4_0 keys and FP16 values. The MTP head accepted a draft token and the supplied F16 projector correctly identified a red image. The host itself rejected Flash Attention off combined with Q8_0 values before starting a worker.

With unified KV enabled, **four concurrently active slots** returned their own ticket identifiers. With three other slots free, one request used **261998 uncached input tokens** from a 262144-token pool and retrieved all three markers. A simultaneous request larger than the remaining budget received HTTP 503. At load, the Q3 four-slot worker peaked at 9.83 GiB WDDM shared usage and 62.58 GiB dedicated. The system committed 99.37 GiB in total (28.38 GiB before this run). Pagefile occupancy peaked at 1.33 GiB.

## Build

Prepare the Windows toolchain listed in [Gufo's build instructions](https://github.com/gufo-org/gufo/blob/v0.35.0/BUILD-WINDOWS.md), including the TheRock gfx1151 SDK, Rust 1.88, CMake, Ninja, vcpkg and the Visual Studio developer environment. In an x64 developer shell run the following command from a clean clone.

```powershell
python build.py .build
```

The script fetches the pinned upstream commit, applies the mixed-format patch and builds the Flash-Next extension in Release mode with HIP device optimization. The output is `.build/build/gufo.exe`. Use `python build.py .build --jobs 8` to set parallelism. For a prepared SDK environment, CMake arguments following a separate `--` replace the default preset.

## Install

Use a new directory for each version. The Windows x64 executable targets AMD Strix Halo (gfx1151) with a compatible graphics driver. Paths to the runtime, dependencies and model shards are chosen on the destination machine.

1. Download `strix-halo-windows-gfx1151.zip` from [this repository's releases](../../releases). The v1.2.1 asset SHA-256 is `b784d5381e3469f09dbb1f0a0677dffd2ec871f415b377bc19580ec5cbe6dfbd`.
2. Extract the complete [upstream Gufo v0.35.0 Windows ZIP](https://github.com/gufo-org/gufo/releases/download/v0.35.0/gufo-v0.35.0-windows-x86_64.zip) into the new directory. Its SHA-256 is `32c76bc1b1c6fda10b6786489fa454b42ec06ef19b1a374100ac95203a6dbbb5`. Keep the DLLs, licenses and kernel directories together.
3. Obtain the following additional DLLs from conda-forge. A `.conda` file is a ZIP containing a `pkg-*.tar.zst` archive. Open that inner archive with a Zstandard-capable archive tool and copy the listed files from `Library/bin/` alongside `gufo.exe`.
4. Extract this project's ZIP over the new directory, allowing it to replace the upstream executable. The `files` object in `engine.json` gives the SHA-256 for each project payload file. Run `.\gufo.exe --version` in that directory. The v1.2.1 package contains native version `0.35.0-strix.3`.
5. Download the GGUF shards separately. Keep the files of each quant together and pass its first shard to `--model`. Supply the separate Q8_0 head to enable MTP.

| Package | Files to copy | Package SHA-256 |
| --- | --- | --- |
| [ICU 79.1](https://conda.anaconda.org/conda-forge/win-64/icu-79.1-h5112557_0.conda) | `icuuc79.dll`, `icudt79.dll` | `f5ffc5548ff51e7fb8949eca909b919ff465401f8d8a79f5fec1d11f9d27b832` |
| [libjpeg-turbo 3.2.1](https://conda.anaconda.org/conda-forge/win-64/libjpeg-turbo-3.2.1-hfd05255_0.conda) | `jpeg8.dll` | `195be7b2ff1eeff02b31cf366437a38d84560c2f6c4facd56b2969024f50957f` |

For example, open PowerShell in the runtime directory and set the two paths for your installation.

```powershell
$firstShard = Read-Host 'Path to the first Flash-Next GGUF shard'
$mtpHead = Read-Host 'Path to the Q8_0 MTP head'
$env:GUFO_FLASH_CACHE_K = 'q8_0'
$env:GUFO_FLASH_CACHE_V = 'q8_0'
.\gufo.exe serve llm --host 127.0.0.1 --port 8080 --context 262144 --sessions 1 --model $firstShard --speculative mtp --mtp-model $mtpHead
```

For autoregressive generation, omit the speculative arguments. To give four sessions one shared window, use `.\gufo.exe serve lmstudio 8080 flash-next -- --model $firstShard --ctx-size 262144 --parallel 4 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --spec-type mtp --spec-draft-model $mtpHead`. Models can stay in their original directories. Rollback consists of stopping this process and launching the executable from the previous, complete runtime directory.

The project additions and Gufo are MIT licensed. Gufo's notices cover its incorporated components, including ggml (MIT) and Rust crates. Its unmodified Symphonia crates use MPL-2.0. Their exact versions and checksums are recorded in [the upstream Cargo.lock](https://github.com/gufo-org/gufo/blob/v0.35.0/Cargo.lock), and sources are available from [crates.io](https://crates.io/crates/symphonia). Full third-party texts accompany the release. Peonist publishes the HaloGen container under its [own EULA](https://github.com/peonist-ai/halogen-flash-server/blob/main/LICENSE.md).

## Measured on Radeon 8060S

With Q3_K_XL, an external Q8_0 MTP head, a single active request and a 262144-token window, a cold C++ retrieval prompt used **262067 actual input tokens at 865.0 prefill tokens/s**. The 27-token answer recovered three independently generated markers. Gufo measured 43.9 decode tokens/s and accepted 22 of 31 MTP proposals on this short answer.

In a separate Hermes tool-and-reasoning test, the model used `read_file` and solved a grid-counting problem with an independently checked answer. Across two requests Hermes counted 15457 reasoning tokens. The longer request had 4461 prompt tokens (4200 cached), generated **15442 tokens at 37.1 tokens/s**, and accepted 11758 of 20470 MTP proposals. The long decode and the long prefill were measured in separate requests.

For Q3_K_XL and IQ4_XS, release-library GPU projections were compared with independent ggml CPU dequantization on real shard rows, with both one and 64 input tokens. The largest Q3 relative L2 error was 0.000041. Q2_0 expansion matched all 10240 sampled reference values.

Two Q8_0 KV runs on v1.1.0 with one session and native prefill sizing processed 262067 and 262084 new tokens at 851.5 and 881.9 tokens/s (arithmetic mean **866.7**). A separate v1.2.0 run honoring the host's physical microbatch of 512 used the four-slot shared pool and reached **561.2 prefill tokens/s** on 261998 new tokens. Its 40-token response decoded at 48.4 tokens/s with 31 of 34 proposals accepted. These are single-request rates. [Machine-readable measurements](measurements.json)

With the released executable, a shared 262144-token window and a 4096-token microbatch, a cold 262033-token request reached **847.4 prefill tokens/s**. The 42-token retrieval answer decoded at 44.8 tokens/s with 32 of 40 MTP proposals accepted. A separate tool-use run verified 4123 reasoning tokens across two requests. The longer response generated 5196 tokens at **42.8 tokens/s**. Its actual prompt was 4520 tokens with 4200 cached.

In matched four-session Q3 loads (262144 tokens, Q8_0 K/V, separate MTP, identical batch settings), peak private process memory was **23.19 GiB** with separate windows and **9.29 GiB** with one shared pool. The executable returned its mapped KV backing to zero after the full-context request completed.

## Check the kernels and package

In the prepared Windows developer environment, run `tests/build-test-dll.ps1 -Work .build -Destination .build/check`. This links the actual release kernel libraries to test exports. Run the following command for the first Q3 or IQ4 shard with NumPy installed. The CPU reference directory must contain an independently built ggml-base.dll exporting the relevant dequantizers, including dequantize_row_q2_0.

```powershell
python tests/mixed_rows.py FIRST_SHARD .build/check/mixed-test.dll GUFO_RUNTIME GGML_REFERENCE .build/gufo
python package.py .build/build/gufo.exe .build/gufo dist/strix-halo-windows-gfx1151.zip
```

The official release executable was built with TheRock 10.0.0, MSVC 14.44 headers and libraries, the Windows 10.0.26100 SDK, clang-cl/lld, Rust 1.88, ICU 79.1 and libjpeg-turbo 3.2.1. Alternative dependency toolchains can change which DLLs a local build needs. The dependency versions used by the published executable are listed under Install.

Run `python tests/kv_rows.py GUFO_RUNTIME GGML_REFERENCE .build/check/mixed-test.dll` for the KV codec comparison. It checks a nonzero row offset, untouched boundary rows, the packed codes, and expansion against ggml. Q8_0 picks the code farther from zero on exact mathematical midpoints. A float32 reciprocal in the CPU reference can choose the equally near code on the other side.
