"""Interactive installer for the motion server backends.

Run with any system python 3.8+, stdlib only:

    python motion_server\\install.py            (menu)
    python motion_server\\install.py ardy       (skip the menu)
    python motion_server\\install.py dart
    python motion_server\\install.py all

Downloads and wires up everything each backend needs (venvs via uv, repo
clones, checkpoints, the NF4 text encoder, DART's google drive data) and
verifies the result. Re-run safe: every step checks what already exists and
skips it. The only parts it cannot automate are the SMPL body model
downloads for DART, those sit behind a registration wall and the script
walks you through them instead.
"""

import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent.resolve()
REPO = HERE.parent

ARDY_GIT = 'https://github.com/nv-tlabs/ardy'
DART_GIT = 'https://github.com/zkf1997/DART'
ENCODER_REPO = 'Aero-Ex/KIMODO-Meta3_llm2vec_NF4'
CORE8_REPO = 'nvidia/ARDY-Core-RP-20FPS-Horizon8'
TORCH_DART = 'torch==2.5.1+cu121'
TORCH_DART_INDEX = 'https://download.pytorch.org/whl/cu121'
TORCH_ARDY_INDEX = 'https://download.pytorch.org/whl/cu126'

# DART checkpoint/data drive folder, linked from the DART repo README
DART_DRIVE_ROOT = '1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps'
# (drive folder path, file name, min size bytes) -> saved to DART/<path>/<name>
DART_FILES = [
    ('mld_denoiser/mld_fps_clip_repeat_euler', 'checkpoint_300000.pt', 70e6),
    ('mld_denoiser/mld_fps_clip_repeat_euler', 'args.yaml', 100),
    ('mld_denoiser/smplh_hml3d_2_8_4', 'checkpoint_300000.pt', 70e6),
    ('mld_denoiser/smplh_hml3d_2_8_4', 'args.yaml', 100),
    ('mvae/mvae_fps_clip', 'checkpoint_200000.pt', 40e6),
    ('mvae/mvae_fps_clip', 'args.yaml', 100),
    ('mvae/mvae_smplh_hml3d_2_8_4', 'checkpoint_200000.pt', 40e6),
    ('mvae/mvae_smplh_hml3d_2_8_4', 'args.yaml', 100),
    ('data/seq_data_zero_male', 'mean_std_h2_f8.pkl', 1000),
    ('data/seq_data_zero_male', 'train_text_embedding_dict.pkl', 9e6),
    ('data/hml3d_smplh/seq_data_zero_male', 'mean_std_h2_f8.pkl', 1000),
    ('data/hml3d_smplh/seq_data_zero_male', 'train_text_embedding_dict.pkl', 50e6),
    ('data', 'stand.pkl', 1000),
    ('data', 'stand_20fps.pkl', 1000),
]

SMPLX_NPZ = 'data/smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_MALE.npz'
SMPLH_MERGED = 'data/smplx_lockedhead_20230207/models_lockedhead/smplh/SMPLH_MALE.pkl'
SMPLH_INPUTS = ['smplh_extract/male/model.npz', 'smplh_extract/female/model.npz',
                'smplh_extract/neutral/model.npz',
                'mano_v1_2/models/MANO_LEFT.pkl', 'mano_v1_2/models/MANO_RIGHT.pkl']


# -- console helpers ----------------------------------------------------------

os.system('')  # enable ansi colors on windows terminals

def _c(code, s):
    return f'\x1b[{code}m{s}\x1b[0m' if sys.stdout.isatty() else str(s)

def head(s):
    print(f'\n{_c("1;36", "==")} {_c("1", s)}')

def ok(s):
    print(f'  {_c("32", "ok")}   {s}')

def skip(s):
    print(f'  {_c("90", "skip")} {s}')

def warn(s):
    print(f'  {_c("33", "warn")} {s}')

def fail(s):
    print(f'  {_c("31", "FAIL")} {s}')
    sys.exit(1)

def ask(prompt, default='y'):
    hint = 'Y/n' if default == 'y' else 'y/N'
    a = input(f'  {prompt} [{hint}] ').strip().lower()
    return (a or default).startswith('y')


def run(cmd, cwd=None, optional=False, quiet=False):
    """run a command, streaming output. returns True on success."""
    show = ' '.join(str(c) for c in cmd)
    if not quiet:
        print(f'  {_c("90", "$ " + show)}')
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if r.returncode != 0 and not optional:
        fail(f'command failed ({r.returncode}): {show}')
    return r.returncode == 0


def capture(cmd, cwd=None):
    try:
        r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                           capture_output=True, text=True)
    except FileNotFoundError:
        return 127, ''
    return r.returncode, (r.stdout or '') + (r.stderr or '')


# -- prerequisites ------------------------------------------------------------

def find_uv():
    local = REPO / 'bin' / 'uv.exe'
    if local.exists():
        return local
    on_path = shutil.which('uv')
    if on_path:
        return Path(on_path)
    fail('uv not found. run the main setup.bat once (it installs bin\\uv.exe) '
         'or install uv from https://docs.astral.sh/uv/')


def check_prereqs():
    head('prerequisites')
    global UV
    UV = find_uv()
    ok(f'uv: {UV}')
    if not shutil.which('git'):
        fail('git not found on PATH, install it from https://git-scm.com')
    ok('git found')
    if shutil.which('curl'):
        ok('curl found')
    else:
        warn('curl not found, falling back to urllib for downloads')
    rc, out = capture(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
    if rc == 0 and out.strip():
        ok(f'gpu: {out.strip().splitlines()[0]}')
    else:
        warn('nvidia-smi not found. both backends need a cuda gpu to run '
             '(installing anyway is fine, e.g. this box only serves files)')


# -- venv + pip ---------------------------------------------------------------

def venv_python(venv):
    return venv / 'Scripts' / 'python.exe'


def ensure_venv(venv, pyver):
    py = venv_python(venv)
    if py.exists():
        skip(f'{venv.name} exists')
    else:
        run([UV, 'venv', venv, '--python', pyver])
        ok(f'{venv.name} created (python {pyver})')
    return py


def pip(venv, *args):
    run([UV, 'pip', 'install', '--python', venv_python(venv), *args])


def py_ok(venv, code):
    rc, _ = capture([venv_python(venv), '-c', code])
    return rc == 0


# -- downloads ----------------------------------------------------------------

def fetch_text(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return urllib.request.urlopen(req).read().decode('utf-8', 'replace')


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    if shutil.which('curl'):
        run(['curl', '-L', '--progress-bar', '-o', tmp, url], quiet=True)
    else:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)


def drive_list(folder_id):
    """list a public google drive folder without gdown: the folder page html
    carries data-id + data-tooltip pairs for every entry. tooltips read
    '<name> Shared folder' for folders and '<name> <type>' for files, so
    folder names are exact and file entries keep the raw tooltip."""
    html = fetch_text(f'https://drive.google.com/drive/folders/{folder_id}')
    out, seen = [], set()
    for m in re.finditer(r'data-id="([-\w]{25,})"[^>]*?data-tooltip="([^"]+)"', html):
        fid, tip = m.group(1), m.group(2)
        if fid in seen:
            continue
        seen.add(fid)
        if tip.endswith(' Shared folder'):
            out.append((fid, tip[: -len(' Shared folder')], True))
        else:
            out.append((fid, tip, False))
    return out


def drive_file(dest, folder_id, name, min_size):
    """download one named file out of a drive folder listing."""
    for fid, tip, is_dir in drive_list(folder_id):
        if not is_dir and (tip == name or tip.startswith(name + ' ')):
            url = f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t'
            download(url, dest)
            with open(dest, 'rb') as f:
                first = f.read(1)
            if dest.stat().st_size < min_size or first == b'<':
                dest.unlink()
                fail(f'{name} download came back wrong (rate limited?), re-run in a bit')
            return True
    return False


def drive_walk(path):
    """resolve a slash path of folder names under the DART drive root."""
    fid = DART_DRIVE_ROOT
    for part in path.split('/'):
        for cid, name, is_dir in drive_list(fid):
            if is_dir and name == part:
                fid = cid
                break
        else:
            return None
    return fid


# -- ARDY backend -------------------------------------------------------------

def install_ardy():
    head('ARDY backend (core8)')
    venv = HERE / '.venv-ardy'
    ensure_venv(venv, '3.11')

    ardy_dir = HERE / 'ARDY'
    if (ardy_dir / 'ardy').exists():
        skip('ARDY clone exists')
    else:
        run(['git', 'clone', '--depth', '1', ARDY_GIT, ardy_dir])
        ok('ARDY cloned')

    if py_ok(venv, 'import ardy'):
        skip('ardy package installed')
    else:
        pip(venv, '-e', ardy_dir)
        ok('ardy installed editable')

    rc, out = capture([venv_python(venv), '-c',
                       'import torch; print(torch.__version__, torch.cuda.is_available())'])
    if rc == 0 and 'cu' in out and 'True' in out:
        skip(f'torch ready ({out.strip()})')
    else:
        pip(venv, 'torch', '--index-url', TORCH_ARDY_INDEX)
        ok('torch (cu126) installed')

    if py_ok(venv, 'import bitsandbytes, websockets'):
        skip('bitsandbytes + websockets installed')
    else:
        pip(venv, 'bitsandbytes', 'websockets')
        ok('bitsandbytes + websockets installed')

    enc = HERE / 'text_encoders' / 'llm2vec_nf4'
    if (enc / 'config.json').exists() and any(enc.glob('*.safetensors')):
        skip('NF4 text encoder present')
    else:
        print('  downloading the NF4 llm2vec encoder (~5.5GB)...')
        run([venv_python(venv), '-c',
             'from huggingface_hub import snapshot_download; '
             f'snapshot_download({ENCODER_REPO!r}, local_dir={str(enc)!r})'])
        ok('NF4 text encoder downloaded')

    if ask('prefetch the core8 checkpoint now (765MB, otherwise first run grabs it)?'):
        run([venv_python(venv), '-c',
             'from huggingface_hub import snapshot_download; '
             f'snapshot_download({CORE8_REPO!r})'])
        ok('core8 checkpoint cached')

    if py_ok(venv, 'from motion_correction import motion_postprocess'):
        ok('motion_correction extension builds')
    else:
        warn('motion_correction (foot skate cleanup ext) not importable. '
             'usually fine, but if generation errors mention it, build with '
             'MinGW inside ARDY/ (see README section 6)')

    print('  verifying imports...')
    rc, out = capture([venv_python(venv), '-c',
                       'import torch, websockets; '
                       'from ardy.model.load_model import load_model; '
                       'from ardy.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder; '
                       "print('cuda', torch.cuda.is_available())"])
    if rc != 0:
        fail(f'ardy venv verification failed:\n{out}')
    ok(f'ARDY backend ready ({out.strip()})')


# -- DART backend -------------------------------------------------------------

def install_dart():
    head('DART backend (babel + hml3d)')
    venv = HERE / '.venv'
    ensure_venv(venv, '3.10')

    if py_ok(venv, 'import smplx, tyro, websockets, clip'):
        skip('requirements installed')
    else:
        pip(venv, '-r', HERE / 'requirements.txt')
        ok('requirements installed')

    rc, out = capture([venv_python(venv), '-c', 'import torch; print(torch.__version__)'])
    if rc == 0 and '+cu121' in out:
        skip(f'torch pinned ({out.strip()})')
    else:
        # must pin the +cu121 local tag or a plain wheel clobbers it
        pip(venv, TORCH_DART, '--index-url', TORCH_DART_INDEX)
        ok('torch 2.5.1+cu121 installed')

    dart_dir = HERE / 'DART'
    if (dart_dir / 'model').exists():
        skip('DART clone exists')
    else:
        run(['git', 'clone', '--depth', '1', DART_GIT, dart_dir])
        ok('DART cloned')

    head('DART checkpoints + data (google drive)')
    folder_cache = {}
    for folder, name, min_size in DART_FILES:
        dest = dart_dir / folder / name
        if dest.exists() and dest.stat().st_size >= min_size:
            skip(f'{folder}/{name}')
            continue
        if folder not in folder_cache:
            folder_cache[folder] = drive_walk(folder)
        fid = folder_cache[folder]
        if fid is None or not drive_file(dest, fid, name, min_size):
            fail(f'could not find {folder}/{name} in the drive folder. the '
                 'layout may have moved, check the DART repo README download '
                 'link and fetch it manually (README section 3)')
        ok(f'{folder}/{name} ({dest.stat().st_size / 1e6:.1f} MB)')

    body_models(dart_dir, venv)

    print('  verifying the full DART import chain (takes a minute)...')
    rc, out = capture([venv_python(venv), HERE / 'check_imports.py'])
    if rc != 0:
        tail = '\n'.join(out.strip().splitlines()[-6:])
        if 'SMPL' in out or 'body' in out.lower():
            warn(f'imports fail on body models (expected if you skipped them):\n{tail}')
        else:
            fail(f'DART verification failed:\n{tail}')
    else:
        ok('DART backend ready')


def body_models(dart_dir, venv):
    """the registration-walled part. loops until placed, merged, or skipped."""
    head('DART body models (registration required, cannot be automated)')
    while True:
        have_x = (dart_dir / SMPLX_NPZ).exists()
        have_h = (dart_dir / SMPLH_MERGED).exists()
        have_h_in = all((dart_dir / p).exists() for p in SMPLH_INPUTS)
        print(f'  babel needs SMPL-X:  {_c("32", "found") if have_x else _c("31", "missing")}'
              f'  DART/{SMPLX_NPZ}')
        print(f'  hml3d needs SMPL-H:  {_c("32", "found") if have_h else _c("31", "missing")}'
              f'  DART/{SMPLH_MERGED}')
        if have_x and have_h:
            ok('both body models in place')
            return
        if not have_h and have_h_in:
            print('  smplh + mano inputs found, building the merged bodies...')
            pip(venv, '--no-deps', 'chumpy')
            run([venv_python(venv), HERE / 'merge_smplh.py'], cwd=HERE)
            continue
        print()
        print('  how to get them (one-time, free registration):')
        if not have_x:
            print(f'''    SMPL-X (babel):
      1. register at https://smpl-x.is.tue.mpg.de
      2. download "SMPL-X with locked head" (smplx_lockedhead)
      3. extract so this file exists:
         {dart_dir / SMPLX_NPZ}''')
        if not have_h:
            print(f'''    SMPL-H (hml3d), SEPARATE registration:
      1. register at https://mano.is.tue.mpg.de
      2. download "Extended SMPL+H" (smplh.tar.xz), extract to:
         {dart_dir}\\smplh_extract\\  (male/female/neutral/model.npz)
      3. download mano_v1_2.zip, put MANO_LEFT.pkl + MANO_RIGHT.pkl in:
         {dart_dir}\\mano_v1_2\\models\\
      4. come back here, the script merges them for you''')
        print()
        a = input('  [r]echeck once files are placed, or [s]kip body models for now: ').strip().lower()
        if a == 's':
            warn('skipped. the server will not start for the affected model '
                 'until the files exist. re-run this installer anytime.')
            return


# -- main ---------------------------------------------------------------------

def summary(chose_ardy, chose_dart):
    head('done. how to run')
    if chose_ardy:
        print(f'  {_c("1", "core8 (best):")} motion_server\\.venv-ardy\\Scripts\\python.exe '
              'motion_server\\server.py --model core8')
    if chose_dart:
        print(f'  {_c("1", "hml3d:")}        motion_server\\.venv\\Scripts\\python.exe '
              'motion_server\\server.py --model hml3d')
        print(f'  {_c("1", "babel:")}        motion_server\\.venv\\Scripts\\python.exe '
              'motion_server\\server.py --model babel')
    print('''
  then in config.yml on the main app box:
    motion:
      enabled: true
      server_host: 127.0.0.1   (or the gpu box ip)
      server_port: 8765

  useful flags: --steps 8 (ardy denoise steps), --history 2.0 (ardy context
  seconds), --guidance 5.0 (dart), --port 8765. full docs in the README.''')


def main():
    print(_c('1;35', '\n  Gabriel motion server installer'))
    print('  models: core8 (ARDY, best quality) / babel + hml3d (DART, original)')

    choice = (sys.argv[1] if len(sys.argv) > 1 else '').lower()
    if choice not in ('ardy', 'dart', 'all'):
        print('''
  what do you want to install?
    1) ARDY core8        best quality, ~7GB disk, 5.7GB vram, zero registration
    2) DART babel+hml3d  the original backend, needs SMPL body model signups
    3) both''')
        raw = input('  choice [1]: ').strip().lower() or '1'
        choice = {'1': 'ardy', '2': 'dart', '3': 'all',
                  'ardy': 'ardy', 'dart': 'dart', 'all': 'all', 'both': 'all'}.get(raw)
        if choice is None:
            fail('unknown choice')

    check_prereqs()
    if choice in ('ardy', 'all'):
        install_ardy()
    if choice in ('dart', 'all'):
        install_dart()
    summary(choice in ('ardy', 'all'), choice in ('dart', 'all'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\naborted, re-run anytime (finished steps are skipped)')
