#!/usr/bin/env python3
"""Compare actual Rust/Nim verify-mode CSV measurements; never infer missing runs.

Shape requirements: matching square sizes and deterministic binary-fraction inputs.
"""
import argparse,csv,json,math,pathlib

def read(path,language):
    """Shape requirements: CSV rows identify kernel and positive square size."""
    rows={}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.startswith(language+','): continue
        values=next(csv.reader([line]));lang,kernel,size,seconds,checksum,error=values
        record=dict(language=lang,kernel=kernel,size=int(size),seconds=float(seconds),checksum=float(checksum),max_error=float(error))
        if record['size']<=0 or record['seconds']<0 or not all(math.isfinite(record[k]) for k in ['seconds','checksum','max_error']): raise ValueError('invalid benchmark row')
        rows[(kernel,int(size))]=record
    if not rows: raise ValueError(f'no {language} benchmark rows in {path}')
    return rows

def main():
    """Shape requirements: every Rust kernel/size must have a Nim counterpart."""
    parser=argparse.ArgumentParser();parser.add_argument('rust');parser.add_argument('nim');parser.add_argument('--output');args=parser.parse_args()
    rust=read(args.rust,'rust');nim=read(args.nim,'nim')
    if rust.keys()!=nim.keys(): raise ValueError(f'unmatched cases: {rust.keys() ^ nim.keys()}')
    comparisons=[]
    for key in sorted(rust,key=lambda k:(k[1],k[0])):
        a,b=rust[key],nim[key]
        if a['max_error']>1e-10 or b['max_error']>1e-10: raise ValueError(f'kernel mismatch {key}')
        if not math.isclose(a['checksum'],b['checksum'],rel_tol=1e-12,abs_tol=1e-9): raise ValueError(f'cross-language checksum mismatch {key}')
        comparisons.append(dict(kernel=key[0],size=key[1],rust_seconds=a['seconds'],nim_seconds=b['seconds'],nim_over_rust=b['seconds']/a['seconds'] if a['seconds'] else None,checksum=a['checksum']))
    text=json.dumps(dict(measurement='single timed verification run, not a confidence interval',comparisons=comparisons),indent=2)
    print(text)
    if args.output: pathlib.Path(args.output).write_text(text+'\n')
if __name__=='__main__':main()
