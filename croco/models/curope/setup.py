# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import shutil
import sys
from pathlib import Path

from setuptools import setup
from torch import cuda
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _unique_existing_paths(paths):
    seen = set()
    results = []
    for path in paths:
        if path is None:
            continue
        resolved = Path(path).expanduser().resolve()
        key = str(resolved)
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        results.append(str(resolved))
    return results


def _find_cuda_home():
    conda_prefix = Path(os.environ.get('CONDA_PREFIX', sys.prefix)).expanduser()
    env_cuda_home = os.environ.get('CUDA_HOME')
    nvcc_path = shutil.which('nvcc')

    candidates = []
    if env_cuda_home:
        candidates.append(Path(env_cuda_home))
    if nvcc_path:
        candidates.append(Path(nvcc_path).resolve().parent.parent)
    candidates.append(conda_prefix)

    for candidate in candidates:
        if candidate.exists() and (candidate / 'bin' / 'nvcc').exists():
            return candidate
    return None


def _find_conda_root(conda_prefix):
    candidates = [
        conda_prefix,
        conda_prefix.parent,
        conda_prefix.parent.parent,
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / 'pkgs').exists():
            return candidate
    return conda_prefix


def _find_cuda_include_dirs():
    conda_prefix = Path(os.environ.get('CONDA_PREFIX', sys.prefix)).expanduser()
    py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    pkgs_dir = _find_conda_root(conda_prefix) / 'pkgs'

    candidates = []
    if pkgs_dir.exists():
        for pattern in (
            'cuda-cudart-dev-*',
            'cuda-cudart-dev_linux-64-*',
            'cuda-cccl-*',
            'cuda-cccl_linux-64-*',
            'cuda-toolkit*',
        ):
            for match in pkgs_dir.glob(pattern):
                candidates.append(match / 'include')
                candidates.append(match / 'targets' / 'x86_64-linux' / 'include')

    candidates.extend([
        conda_prefix / 'include',
        conda_prefix / 'targets' / 'x86_64-linux' / 'include',
        conda_prefix / 'lib' / py_ver / 'site-packages' / 'nvidia' / 'cuda_runtime' / 'include',
        conda_prefix / 'lib' / py_ver / 'site-packages' / 'triton' / 'backends' / 'nvidia' / 'include',
    ])

    existing = []
    for path in candidates:
        if (path / 'cuda_runtime.h').exists() or (path / 'nv' / 'target').exists():
            existing.append(path)
            if (path / 'cccl').exists():
                existing.append(path / 'cccl')
    return _unique_existing_paths(existing)


def _find_cuda_library_dirs():
    conda_prefix = Path(os.environ.get('CONDA_PREFIX', sys.prefix)).expanduser()
    pkgs_dir = _find_conda_root(conda_prefix) / 'pkgs'

    candidates = [
        conda_prefix / 'lib',
        conda_prefix / 'targets' / 'x86_64-linux' / 'lib',
    ]
    if pkgs_dir.exists():
        for pattern in ('cuda-cudart-*', 'cuda-cudart_linux-64-*', 'cuda-cudart-dev-*', 'cuda-cudart-dev_linux-64-*'):
            for match in pkgs_dir.glob(pattern):
                candidates.append(match / 'lib')
                candidates.append(match / 'targets' / 'x86_64-linux' / 'lib')

    existing = []
    for path in candidates:
        if any(path.glob('libcudart.so*')):
            existing.append(path)
    return _unique_existing_paths(existing)


def _ensure_cudart_link(library_dirs):
    for directory in library_dirs:
        link_path = Path(directory) / 'libcudart.so'
        if link_path.exists() and link_path.resolve().exists():
            return library_dirs

    for directory in library_dirs:
        candidates = sorted(Path(directory).glob('libcudart.so.*'))
        candidates = [candidate for candidate in candidates if candidate.exists()]
        if not candidates:
            continue
        stub_dir = Path(__file__).resolve().parent / '.cuda_libs'
        stub_dir.mkdir(exist_ok=True)
        stub_link = stub_dir / 'libcudart.so'
        if stub_link.exists() or stub_link.is_symlink():
            stub_link.unlink()
        stub_link.symlink_to(candidates[0])
        return [str(stub_dir)] + library_dirs

    return library_dirs


cuda_home = _find_cuda_home()
if cuda_home is not None:
    os.environ['CUDA_HOME'] = str(cuda_home)
    cpp_extension.CUDA_HOME = str(cuda_home)

cuda_include_dirs = _find_cuda_include_dirs()
cuda_library_dirs = _ensure_cudart_link(_find_cuda_library_dirs())

if not cuda_include_dirs:
    raise RuntimeError(
        'Unable to locate cuda_runtime.h. Install CUDA headers into the active env '
        'or set CUDA_HOME to a valid toolkit root.'
    )

print(f'Using CUDA_HOME={os.environ.get("CUDA_HOME", "")}')
print(f'Using CUDA include dirs={cuda_include_dirs}')
print(f'Using CUDA library dirs={cuda_library_dirs}')

# compile for all possible CUDA architectures
all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()
# alternatively, you can list cuda archs that you want, eg:
# all_cuda_archs = [
    # '-gencode', 'arch=compute_70,code=sm_70',
    # '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_80,code=sm_80',
    # '-gencode', 'arch=compute_86,code=sm_86'
# ]

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                include_dirs=cuda_include_dirs,
                library_dirs=cuda_library_dirs,
                extra_compile_args = dict(
                    nvcc=['-O3','--ptxas-options=-v',"--use_fast_math"]+all_cuda_archs, 
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
