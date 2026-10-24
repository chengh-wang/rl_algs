"""Fetch the pinned public example assets; keep motion data outside the package."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import runpy
import urllib.request

REPO='lvhaidong/LAFAN1_Retargeting_Dataset'
REVISION='ce1572906efe6157840e8474d5a0d7aa87481e74'
ROOT=Path(__file__).resolve().parents[1]


def main():
    info=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}'))
    assert info['sha']==REVISION
    files=[]
    for row in info['siblings']:
        name=row['rfilename']
        if name.startswith('g1/') and name.endswith('.csv'):
            files.append((name,ROOT/'data/lafan1_g1'/Path(name).name))
        elif name.startswith('robot_description/g1/'):
            files.append((name,ROOT/'data/g1_description'/name.removeprefix('robot_description/g1/')))
        elif name in ('LICENSE','README.md'):
            files.append((name,ROOT/'data/source_notices'/name))

    def fetch(pair):
        name,path=pair; path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            payload=urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}').read()
            temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_bytes(payload); temporary.replace(path)
        return dict(source=name,local=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=8) as pool: records=list(pool.map(fetch,files))
    (ROOT/'data/source_manifest.json').write_text(json.dumps(dict(repo=REPO,revision=REVISION,files=records),indent=2)+'\n')
    runpy.run_path(str(ROOT/'scripts/urdf_to_mjcf.py'),run_name='__main__')
    print(f'Prepared {len(records)} pinned assets; manifest: data/source_manifest.json')


if __name__=='__main__': main()
