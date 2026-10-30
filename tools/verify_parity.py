"""Compare public Nim/Rust fixture exports with each other and NumPy (dev only)."""
import json, sys
from pathlib import Path
import numpy as np
rust=json.loads(Path(sys.argv[1]).read_text())
nim=json.loads(Path(sys.argv[2]).read_text())
assert len(rust)==len(nim)
for a,b in zip(rust,nim):
    assert a['name']==b['name'] and a['shape']==b['shape'],(a,b)
    np.testing.assert_allclose(a['data'],b['data'],rtol=1e-8,atol=1e-9,err_msg=a['name'])
    if a['name'].startswith(('broadcast_','transpose_','slice_')):
        op,m,n=a['name'].split('_');m,n=int(m),int(n)
        source=np.arange(2*m*n,dtype=np.float64).reshape(2,m,n)
        expected={'broadcast':lambda:source+np.arange(1,n+1).reshape(1,n),'transpose':lambda:source.transpose(2,0,1),'slice':lambda:source[:,:,::2]}[op]()
        assert list(expected.shape)==a['shape']
        np.testing.assert_array_equal(a['data'],expected.ravel(),err_msg=a['name'])
    else:
        expected={'vector_norms':[25,7,5],'cross':[0,0,1], 'inverse':np.linalg.inv([[0,2,3],[4,5,6],[7,8,10]]).ravel(), 'determinant':[-5], 'svd':np.linalg.svd([[1,2],[2,4]],compute_uv=False),'eigen':np.linalg.eigvalsh([[2,1],[1,2]])}[a['name']]
        np.testing.assert_allclose(a['data'],expected,rtol=1e-8,atol=1e-9)
print(f'PASS: {len(rust)} Nim/Rust cases, independently checked with NumPy {np.__version__}')
