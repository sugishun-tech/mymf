#!/usr/bin/env python3
"""Regenerate width-generic register kernels. Run from any directory."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
parts=['''/* Register accumulators for short dense rows. Generated cases differ only
   in vector count; they do not recognize datasets, ranks, or seeds. The final
   vector is masked so no load can cross a row or buffer boundary. */
#ifndef MYMF_REG_H
#define MYMF_REG_H
#if MYMF_RUNTIME_X86
#include <immintrin.h>
''']
for isa in ['avx512', 'avx2']:
 for dtype,c_type,suffix,lanes,maskbits in [('f32','float','ps',16 if isa=='avx512' else 8,32),('f64','double','pd',8 if isa=='avx512' else 4,64)]:
  for idx in ['i32','i64']:
   widthbits = 512 if isa=='avx512' else 256
   vec=f'__m{widthbits}'+('d' if dtype=='f64' else '')
   pre=f'_mm{widthbits}_'
   target='MYMF_AVX512' if isa=='avx512' else 'MYMF_AVX2'
   limit=lanes*8
   name=f'reg_{dtype}_{idx}_{isa}'
   base=f'row_{dtype}_{idx}_{isa}'
   parts.append(f'''{target} static void {name}(const {c_type} * MYMF_RESTRICT data,
    const int{idx[1:]}_t * MYMF_RESTRICT indices, size_t begin, size_t end,
    const {c_type} * MYMF_RESTRICT b, {c_type} * MYMF_RESTRICT out, size_t k) {{
    if (k > {limit}) {{ {base}(data, indices, begin, end, b, out, k); return; }}
    const size_t last = (k - 1) % {lanes} + 1;
''')
   if isa=='avx512':
    parts.append(f'    const __mmask{lanes} mask = (__mmask{lanes})((1u << last) - 1u);\n')
    maskload=lambda addr: f'{pre}maskz_loadu_{suffix}(mask, {addr})'
    maskstore=lambda addr,v: f'{pre}mask_storeu_{suffix}({addr}, mask, {v})'
   else:
    setop=f'{pre}setr_epi{maskbits}'
    if maskbits==64: setop=f'{pre}setr_epi64x'
    parts.append(f'    const __m256i mask = {setop}('+', '.join(f'last > {j} ? -1 : 0' for j in range(lanes))+');\n')
    maskload=lambda addr: f'{pre}maskload_{suffix}({addr}, mask)'
    maskstore=lambda addr,v: f'{pre}maskstore_{suffix}({addr}, mask, {v})'
   parts.append(f'    switch ((k + {lanes-1}) / {lanes}) {{\n')
   for c in range(1,9):
    parts.append(f'    case {c}: {{\n')
    for j in range(c): parts.append(f'        {vec} s{j} = {pre}setzero_{suffix}();\n')
    parts.append(f'''        for (size_t p = begin; p < end; ++p) {{
            const {vec} a = {pre}set1_{suffix}(data[p]);
            const {c_type} *x = b + (size_t)indices[p] * k;
''')
    for j in range(c):
     v=maskload(f'x + {j*lanes}') if j==c-1 else f'{pre}loadu_{suffix}(x + {j*lanes})'
     parts.append(f'            s{j} = {pre}fmadd_{suffix}(a, {v}, s{j});\n')
    parts.append('        }\n')
    for j in range(c):
     v=maskstore(f'out + {j*lanes}',f's{j}') if j==c-1 else f'{pre}storeu_{suffix}(out + {j*lanes}, s{j})'
     parts.append(f'        {v};\n')
    parts.append('        return;\n    }\n')
   parts.append('    }\n}\n')
parts.append('#endif\n#endif\n')
(root/'src/mymf/_reg.h').write_text(''.join(parts))
