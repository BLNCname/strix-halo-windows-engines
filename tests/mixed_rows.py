"""Compare the candidate's GPU kernels on real, packed GGUF rows with the Strix ggml CPU decoder."""

import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import sys

import numpy as np


def verify(args):
    sys.path.insert(0, str(args.gufo_source / 'tools' / 'gufo'))
    from gguf import parse_gguf
    directory = os.add_dll_directory(str(args.runtime))
    hip = ct.CDLL(str(args.runtime / 'amdhip64_7.dll'))
    native = ct.CDLL(str(args.test_dll))
    cpu = ct.CDLL(str(args.strix / 'ggml-base.dll'))
    ptr, integer, size = ct.c_void_p, ct.c_int, ct.c_size_t

    def bind(fn, types):
        fn.argtypes = types
        return fn

    def check(rc):
        if rc != 0:
            raise RuntimeError(f'GPU call failed ({rc})')

    malloc = bind(hip.hipMalloc, [ct.POINTER(ptr), size])
    copy = bind(hip.hipMemcpy, [ptr, ptr, size, integer])
    free = bind(hip.hipFree, [ptr])
    expand = bind(native.iq4_test_expand, [integer, ptr, ptr, size])
    projection = [integer, ptr, ptr, ptr, ptr, integer, integer, integer, integer, integer, ptr]
    matrix = bind(native.qfn_mmq_mixed_moe_raw, projection)
    vector = bind(native.qfn_mmq_moe_vec, projection + [ptr, ptr])
    check(bind(native.qfn_mmq_init, [integer])(0))
    rng = np.random.default_rng(8193)
    codecs = {6: ('q5_0', 32, 22), 14: ('q6_K', 256, 210), 18: ('iq3_xxs', 256, 98),
              20: ('iq4_nl', 32, 18), 22: ('iq2_s', 256, 82), 23: ('iq4_xs', 256, 136),
              42: ('q2_0', 64, 18)}
    needed = {(6, True), (18, True), (20, True), (22, True), (23, True),
              (42, True), (14, False), (23, False)}
    results = []
    for path in sorted(args.model.parent.glob(args.model.name.replace('00001-of-', '*-of-'))):
        header = parse_gguf(path)
        for t in header['tensors']:
            fmt, expert = t['type'], 'exps.weight' in t['name']
            if (fmt, expert) not in needed:
                continue
            needed.remove((fmt, expert))
            name, block, encoded = codecs[fmt]
            print(f'Checking {name}: {t["name"]}', file=sys.stderr, flush=True)
            cols, rows = t['shape'][-1], 8
            stride = cols // block * encoded
            with path.open('rb') as f:
                chunks = []
                for e in range(2):
                    first = e * (t['shape'][-2] * (17 if expert else 0)) + 23 + e * 41
                    f.seek(header['data_offset'] + t['offset'] + first * stride)
                    chunks.append(f.read(rows * stride))
            packed = np.frombuffer(b''.join(chunks), dtype=np.uint8)
            ref = np.empty((2, rows, cols), dtype=np.float32)
            fn = bind(getattr(cpu, 'dequantize_row_' + name), [ptr, ptr, ct.c_int64])
            fn.restype = None
            fn(packed.ctypes.data, ref.ctypes.data, ref.size)
            allocated = []

            def device(arr):
                p = ptr()
                check(malloc(ct.byref(p), arr.nbytes + 4096))
                allocated.append(p)
                check(copy(p, arr.ctypes.data, arr.nbytes, 1))
                return p

            try:
                weights = device(packed)
                row = {'tensor': t['name'], 'format': name}
                if not expert or fmt == 42:
                    data = (np.empty((ref.size // 32, 34), np.uint8) if fmt == 42
                            else np.empty(ref.shape, np.float16))
                    destination = device(data)
                    check(expand(fmt, weights, destination, ref.size))
                    check(copy(data.ctypes.data, destination, data.nbytes, 2))
                    if fmt == 42:
                        vals = data[:, 2:].view(np.int8) * data[:, :2].copy().view(np.float16).astype(np.float32)
                        np.testing.assert_array_equal(vals.reshape(ref.shape), ref)
                    else:
                        expected = ref.astype(np.float16)
                        row['signed_zero_differences'] = int(np.count_nonzero(
                            (data == 0) & (expected == 0) & (np.signbit(data) != np.signbit(expected))))
                        np.testing.assert_array_equal(data, expected)
                    row['equal_weights'] = ref.size
                else:
                    for n in (1, 64):
                        print(f'  {n} token(s)', file=sys.stderr, flush=True)
                        x = rng.integers(-127, 128, (n, cols), dtype=np.int16).astype(np.float32) / 128
                        x[:, ::32] = 127 / 128
                        ids = (np.arange(n, dtype=np.int32) + 1) % 2
                        expected = np.einsum('nmk,nk->nm', ref[ids].astype(np.float64), x)
                        out = np.empty((n, rows), dtype=np.float32)
                        xp, ip, op = device(x), device(ids), device(out)
                        call = (fmt, weights, xp, ip, op, rows, cols, n, 2, 1, None)
                        check(vector(*call, None, None) if n == 1 else matrix(*call))
                        check(copy(out.ctypes.data, op, out.nbytes, 2))
                        rel = np.linalg.norm(out - expected) / np.linalg.norm(expected)
                        if not np.isfinite(rel) or rel > 0.003:
                            raise ValueError(f'{name} at {n} tokens: relative L2 error {rel:.6g}')
                        row['vector_rel_l2' if n == 1 else 'matrix_rel_l2'] = float(rel)
                results.append(row)
            finally:
                for p in allocated:
                    check(free(p))
    directory.close()
    if needed:
        raise ValueError(f'Missing test formats: {needed}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('model', 'test_dll', 'runtime', 'strix', 'gufo_source'):
        parser.add_argument(k, type=Path)
    print(json.dumps(verify(parser.parse_args()), indent=2))
