"""Download St4RTrack checkpoints from Hugging Face into a local directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_REPO_ID = 'yupengchengg147/St4RTrack'



def _download_seq(repo_id: str, output_dir: Path, force_download: bool) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=['Pair/*'],
        force_download=force_download,
    )



def _download_pair(repo_id: str, output_dir: Path, force_download: bool) -> None:
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    downloads = {
        'config.json': 'Pair/config.json',
        'model.safetensors': 'Pair/model.safetensors',
    }
    for local_name, remote_name in downloads.items():
        downloaded = hf_hub_download(repo_id=repo_id, filename=remote_name, force_download=force_download)
        shutil.copy2(downloaded, output_dir / local_name)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download a St4RTrack Hugging Face checkpoint locally.')
    parser.add_argument('--repo_id', default=DEFAULT_REPO_ID)
    parser.add_argument('--variant', default='seq', choices=['seq', 'pair'])
    parser.add_argument('--output_dir', default=None, help='Directory to store the local checkpoint.')
    parser.add_argument('--force_download', action='store_true', default=False)
    return parser.parse_args()



def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir or (Path(__file__).resolve().parent / 'checkpoints' / f'st4rtrack_{args.variant}'))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.variant == 'seq':
        _download_seq(args.repo_id, output_dir, args.force_download)
    else:
        _download_pair(args.repo_id, output_dir, args.force_download)
    print(f'Downloaded {args.variant} checkpoint to {output_dir}')
    print(f'Use it with: --weights {output_dir}')


if __name__ == '__main__':
    main()
