// Test DLL linked against the candidate's compiled kernels. Dispatch by GGML type.
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

extern "C" __declspec(dllexport) int iq4_test_expand(int type, const void* src,
                                                     void* dst, size_t count) {
  using namespace gufo::models::qwen38_flash_next::rocm;
  UploadConversion kind;
  switch (type) {
    case 14: kind = UploadConversion::kDenseQ6; break;
    case 23: kind = UploadConversion::kEmbedIQ4; break;
    case 42: kind = UploadConversion::kExpertQ2; break;
    default: return -1;
  }
  ConvertMixedWeights(kind, src, dst, count, nullptr);
  return hipGetLastError();
}

extern "C" __declspec(dllexport) int test_kv_rows(int bits, int expand,
    const void* input, void* output, unsigned width, unsigned rows,
    const unsigned* position) {
  using namespace gufo::models::qwen38_flash_next;
  const auto format = bits == 8 ? KvType::Q8_0 : KvType::Q4_0;
  const auto* src = static_cast<const __half*>(input);
  auto* dst = static_cast<__half*>(output);
  if (expand) rocm::ExpandCache(src, dst, format, width, rows, position, nullptr);
  else rocm::PackCache(src, dst, format, width, rows, position, nullptr);
  return hipGetLastError();
}
