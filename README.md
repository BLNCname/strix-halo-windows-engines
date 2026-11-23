huge credit goes to [@wenis](https://github.com/wenis), [Peonist](https://github.com/peonist-ai) and the [HaloGen Flash Server project](https://github.com/peonist-ai/halogen-flash-server) for their work on Flash-Next inference on Strix Halo.

# Strix Halo Windows engines

Three selectable ROCm profiles run the standard Unsloth Qwen3.8-Flash-Next GGUF shards on Windows 11 and Radeon 8060S. They share a native executable built from [Gufo v0.35.0](https://github.com/gufo-org/gufo/tree/v0.35.0). Gufo implements the model, loader and server. Its quantized matrix kernels incorporate code from [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp). HaloGen is an inspiration for this project. The executable builds from Gufo and the patch in this repository.

| Profile | Routed expert matrices in the four Unsloth shards |
| --- | --- |
| Q4_K_XL | 94 Q4_K, 43 Q5_1, 5 Q8_0, 2 Q5_K |
| IQ4_XS | 56 IQ3_XXS, 34 IQ4_XS, 33 Q5_0, 13 IQ4_NL, 6 IQ2_S, 2 Q2_0 |
| Q3_K_XL | 76 IQ3_XXS, 26 IQ4_NL, 20 Q5_0, 12 IQ4_XS, 8 IQ2_S, 2 Q2_0 |

The loader dispatches by tensor type. Both mixed profiles retain 142 packed expert matrices and expand two Q2_0 matrices to Q8_0 exactly on upload. Dense Q6_K and the IQ4_XS token embedding expand to FP16. The IQ4_NL per-layer embedding stays host resident. The native context is 262144 tokens. The optional external Q8_0 MTP head is a separate GGUF. Obtain weights from [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

## Build

Prepare the Windows toolchain listed in [Gufo's build instructions](https://github.com/gufo-org/gufo/blob/v0.35.0/BUILD-WINDOWS.md), including the TheRock gfx1151 SDK, Rust 1.88, CMake, Ninja, vcpkg and the Visual Studio developer environment. In an x64 developer shell run the following command from a clean clone.

```powershell
python build.py .build
```

The script fetches the pinned upstream commit, applies the mixed-format patch and builds the Flash-Next extension in Release mode with HIP device optimization. The output is `.build/build/gufo.exe`. For a prepared SDK environment, additional CMake configuration arguments after the work directory replace the default preset.

## Install

The [runtime manager](https://github.com/BLNCname/strix-halo-runtime-manager) installs the release bundle by URL and fetches the matching runtime dependencies from their publishers. The engine package contains the native executable, profile manifest and license notices. Model shards stay where you downloaded them.

The project additions and Gufo are MIT licensed. Gufo's upstream notices cover its incorporated components. Full third-party texts accompany the release.
