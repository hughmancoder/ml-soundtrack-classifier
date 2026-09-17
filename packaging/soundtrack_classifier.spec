# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

repo_root = os.path.abspath(os.path.join(os.path.dirname(SPEC), '..'))

datas = []
datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')
datas += collect_data_files('safehttpx')
datas += collect_data_files('groovy')
datas += collect_data_files('imageio_ffmpeg')
datas += collect_data_files('librosa')
datas += collect_data_files('transformers')
datas += collect_data_files('huggingface_hub')
datas += collect_data_files('tokenizers')

# Include model weights and configuration
config_src = os.path.join(repo_root, 'config', 'config.yaml')
if os.path.exists(config_src):
    datas.append((config_src, 'config'))

models_dir = os.path.join(repo_root, 'models')
if os.path.exists(models_dir):
    for f in os.listdir(models_dir):
        if f.endswith('.pt'):
            datas.append((os.path.join(models_dir, f), 'models'))

# Filter out non-existent files, __pycache__, and numba jit caches
datas = [
    (src, dst) for (src, dst) in datas
    if os.path.exists(src) and '__pycache__' not in src and not src.endswith(('.nbc', '.nbi'))
]

hiddenimports = []
hiddenimports += collect_submodules('gradio')
hiddenimports += collect_submodules('gradio_client')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('starlette')
hiddenimports += collect_submodules('pydantic')
hiddenimports += collect_submodules('soundtrack_classifier')
hiddenimports += collect_submodules('scipy.stats')
hiddenimports += collect_submodules('transformers')
hiddenimports += collect_submodules('huggingface_hub')
hiddenimports += [
    'imageio_ffmpeg',
    'torch',
    'torchaudio',
    'torch._numpy',
    'torch._numpy._ufuncs',
    'scipy',
    'scipy.signal',
    'soundfile',
    'librosa',
    'pandas',
    'yaml',
    'transformers',
    'huggingface_hub',
    'tokenizers',
    'pkg_resources.extern',
]

binaries = []
binaries += collect_dynamic_libs('torch')
binaries += collect_dynamic_libs('torchaudio')
binaries += collect_dynamic_libs('imageio_ffmpeg')

a = Analysis(
    [os.path.join(repo_root, 'src', 'soundtrack_classifier', 'gui.py')],
    pathex=[
        os.path.join(repo_root, 'src'),
        repo_root
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'wandb', 'pytest', 'torchvision'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SoundtrackClassifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SoundtrackClassifier',
)

app = BUNDLE(
    coll,
    name='SoundtrackClassifier.app',
    icon=None,
    bundle_identifier='com.soundtrack.classifier',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': 'True',
        'LSMinimumSystemVersion': '12.0',
    },
)
