#!/usr/bin/env python3
"""Download and safely unpack the public, checksum-verified historical archive."""
from pathlib import Path
import argparse,hashlib,json,shutil,tempfile,urllib.request,zipfile,ssl
import certifi
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'artifacts');args=ap.parse_args()
 if args.out.exists():ap.error('Output directory exists; choose a NEW path')
 meta=json.loads((ROOT/'docs/archive_asset.json').read_text())
 args.out.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=args.out.parent) as work:
  tmp=Path(work);zpath=tmp/meta['name'];req=urllib.request.Request(meta['url'],headers={'User-Agent':'AtomAgent-Bench-reproduction'})
  h=hashlib.sha256();n=0
  with urllib.request.urlopen(req,timeout=120,context=ssl.create_default_context(cafile=certifi.where())) as res,zpath.open('wb') as f:
   while block:=res.read(1024*1024):f.write(block);h.update(block);n+=len(block)
  assert n==meta['size'] and h.hexdigest()==meta['sha256'],'Downloaded archive checksum mismatch'
  unpack=tmp/'unpack';unpack.mkdir()
  with zipfile.ZipFile(zpath) as z:
   for i in z.infolist():
    p=(unpack/i.filename).resolve()
    assert p.is_relative_to(unpack.resolve()),'Unsafe archive path'
    assert not ((i.external_attr>>16)&0o170000)==0o120000,'Archive contains symlink'
   z.extractall(unpack)
  shutil.move(str(unpack),str(args.out))
 print('Verified and extracted',n,'bytes to',args.out)
if __name__=='__main__':main()
