"""Compare HIP KV writes and reads to ggml's reference codecs, including a nonzero row offset."""
import argparse
import ctypes as ct
import os
from pathlib import Path
import numpy as np


def verify(runtime, reference, test_dll):
    with os.add_dll_directory(str(runtime)):
        hip = ct.CDLL(str(runtime / "amdhip64_7.dll"))
        base = ct.CDLL(str(reference / "ggml-base.dll"))
        native = ct.CDLL(str(test_dll))
        ptr, size = ct.c_void_p, ct.c_size_t
        hip.hipMalloc.argtypes = [ct.POINTER(ptr), size]
        hip.hipMemcpy.argtypes = [ptr, ptr, size, ct.c_int]
        hip.hipFree.argtypes = [ptr]
        native.test_kv_rows.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ct.c_uint, ct.c_uint, ptr]
        def check(code):
            if code: raise RuntimeError(f"HIP failure {code}")
        alloc = []
        def upload(array):
            address = ptr()
            check(hip.hipMalloc(ct.byref(address), array.nbytes))
            alloc.append(address)
            check(hip.hipMemcpy(address, array.ctypes.data, array.nbytes, 1))
            return address
        try:
            offset, rows, width = 3, 19, 512
            n = offset + rows + 2
            values = np.random.default_rng(8293).normal(size=(n, width)).astype(np.float16)
            values[3:5] = 0
            source = upload(values)
            position = upload(np.array([offset], dtype=np.uint32))
            for bits, block in [(4, 18), (8, 34)]:
                packed = np.full((n, width // 32 * block), 0x55, dtype=np.uint8)
                device = upload(packed)
                expected = packed.copy()
                ref = getattr(base, f"quantize_row_q{bits}_0_ref")
                ref.argtypes = [ptr, ptr, ct.c_int64]
                reference_input = values[offset:offset + rows].astype(np.float32)
                ref(reference_input.ctypes.data, expected[offset:].ctypes.data, reference_input.size)
                check(native.test_kv_rows(bits, 0, source, device, width, rows, position))
                check(hip.hipMemcpy(packed.ctypes.data, device, packed.nbytes, 2))
                # The sign bit on zero scales is immaterial to both codecs.
                compare = packed.copy().reshape(-1, block)
                exp = expected.copy().reshape(-1, block)
                for a in (compare, exp):
                    a[(a[:, 0] == 0) & ((a[:, 1] & 127) == 0), 1] = 0
                if bits == 8:
                    np.testing.assert_array_equal(compare[:, :2], exp[:, :2])
                    data = reference_input.astype(np.float64).reshape(-1, 32)
                    maxima = np.abs(data).max(axis=1, keepdims=True)
                    ratio = np.divide(data * 127, maxima, out=np.zeros_like(data), where=maxima != 0)
                    nearest = (np.sign(ratio) * np.floor(abs(ratio) + .5)).astype(np.int8)
                    codes = compare[:, 2:].view(np.int8)
                    low, high = offset * width // 32, (offset + rows) * width // 32
                    np.testing.assert_array_equal(codes[low:high], nearest)
                    np.testing.assert_array_equal(compare[:low], exp[:low])
                    np.testing.assert_array_equal(compare[high:], exp[high:])
                    # The CPU rounds after a float32 reciprocal and multiply.
                    # It may choose the other equally near code at an exact midpoint.
                    difference = codes[low:high].astype(int) - exp[low:high, 2:].view(np.int8)
                    ties = difference != 0
                    assert np.all(np.abs(difference[ties]) == 1)
                    assert np.all(np.mod(np.abs(ratio[ties]), 1) == .5)
                    print(f"Q8_0 equal-distance CPU rounding differences: {ties.sum()}")
                else:
                    np.testing.assert_array_equal(compare, exp)
                target = np.full_like(values, -55)
                output = upload(target)
                check(native.test_kv_rows(bits, 1, device, output, width, rows, position))
                check(hip.hipMemcpy(target.ctypes.data, output, target.nbytes, 2))
                dequant = getattr(base, f"dequantize_row_q{bits}_0")
                dequant.argtypes = [ptr, ptr, ct.c_int64]
                expected_float = np.empty((offset + rows, width), dtype=np.float32)
                dequant(packed.ctypes.data, expected_float.ctypes.data, expected_float.size)
                np.testing.assert_array_equal(target[:offset + rows], expected_float.astype(np.float16))
                assert (target[offset + rows:] == -55).all()
                print(f"Q{bits}_0: {reference_input.size} packed values and {expected_float.size} expanded values verified.")
        finally:
            for p in alloc: check(hip.hipFree(p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("runtime", "reference", "test_dll"): parser.add_argument(arg, type=Path)
    verify(**vars(parser.parse_args()))
