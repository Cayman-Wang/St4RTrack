"""Wrapper around St4RTrack inference for static-mesh normalization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from static_mesh.st4rtrack_adapter import normalize_st4rtrack_outputs



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export normalized St4RTrack pair inputs for static mesh.')
    parser.add_argument('--input_dir', required=True, help='Image directory, video path, or npz input accepted by St4RTrack.')
    parser.add_argument('--output_dir', required=True, help='Normalized export root.')
    parser.add_argument('--seq_name', required=True, help='Sequence identifier.')
    parser.add_argument('--existing_infer_dir', default=None, help='Use existing infer output directory instead of rerunning infer.py.')
    parser.add_argument('--weights', default=None, help='Local weights or local Hugging Face snapshot directory.')
    parser.add_argument('--hf_model', default=None, help='Optional Hugging Face repo id, e.g. yupengchengg147/St4RTrack.')
    parser.add_argument('--hf_variant', default='seq', choices=['seq', 'pair'], help='Hugging Face checkpoint variant when --hf_model is used.')
    parser.add_argument('--hf_force_download', action='store_true', default=False, help='Force fresh Hugging Face download when --hf_model is used.')
    parser.add_argument('--infer_script', default=str(Path(__file__).resolve().parent / 'infer.py'))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=200)
    parser.add_argument('--fps', type=int, default=0)
    parser.add_argument('--mid_anchor', action='store_true', default=False)
    parser.add_argument('--dynamic_prior_mode', default='off', choices=['off', 'pointodyssey_instance_motion'])
    parser.add_argument('--mask_root', default=None, help='Optional mask directory override for pair sidecar generation.')
    parser.add_argument('extra_infer_args', nargs='*')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = normalize_st4rtrack_outputs(
        input_path=args.input_dir,
        output_dir=args.output_dir,
        sequence_id=args.seq_name,
        existing_infer_dir=args.existing_infer_dir,
        infer_script=args.infer_script,
        weights=args.weights,
        hf_model=args.hf_model,
        hf_variant=args.hf_variant,
        hf_force_download=args.hf_force_download,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        start_frame=args.start_frame,
        step_size=args.step_size,
        num_frames=args.num_frames,
        fps=args.fps,
        mid_anchor=args.mid_anchor,
        dynamic_prior_mode=args.dynamic_prior_mode,
        mask_root=args.mask_root,
        extra_infer_args=args.extra_infer_args,
    )
    print(f'Saved normalized manifest to {Path(args.output_dir) / "manifest.json"}')
    print(f'- sequence: {manifest.sequence_id}')
    print(f'- frames: {len(manifest.frame_ids)}')
    print(f'- pairs: {len(manifest.pairs)}')
    print(f'- anchor frame: {manifest.anchor_frame_id}')


if __name__ == '__main__':
    main()
