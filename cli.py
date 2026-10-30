from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from alina.config import PipelineConfig, EvaluationConfig
from alina.pipeline import process_image, run_batch
from alina.roi import select_roi
from eval.cbem import create_cbem
from eval.superimpose import overlay_coords
from eval.evaluate import run_evaluation


def _cmd_label(args: argparse.Namespace) -> None:
    if not args.image and not args.input_dir:
        raise SystemExit("alina label: provide either --input-dir (batch mode) or --image (single-image mode)")

    config = PipelineConfig(
        input_dir=args.input_dir or ".",
        output_images_dir=args.output_images_dir,
        output_coords_dir=args.output_coords_dir,
        log_file=args.log_file,
        peak_pixel_threshold=args.peak_threshold,
        min_white_pixels=args.min_white_pixels,
        circular_threshold=args.circular_threshold,
        yellow_lower=tuple(args.yellow_lower),
        yellow_upper=tuple(args.yellow_upper),
        mask_ignore_left_columns=args.mask_ignore_left_columns,
        fast_traversal=args.fast_traversal,
    )

    if args.image:
        image_path = Path(args.image)
        print("Draw the ROI: click Bottom-Left, Top-Left, Top-Right, Bottom-Right, then press any key.")
        roi_points = select_roi(str(image_path))
        img = cv2.imread(str(image_path))
        annotated, coords, had_lines = process_image(img, roi_points, config, thresholder=thresholder)

        cv2.imwrite(str(config.output_images_dir / image_path.name), annotated)
        (config.output_coords_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(f"{x:6d}{y:6d}" for x, y in coords)
        )
        print(f"{image_path.name}: {'Labeled' if had_lines else 'No lines found'}")
        if args.show:
            from alina.roi import compute_display_scale
            scale = compute_display_scale(annotated, max_display_dim=1080)
            display_annotated = cv2.resize(annotated, None, fx=scale, fy=scale) if scale != 1.0 else annotated
            cv2.imshow("Result", display_annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    print("Hello! Welcome to ALINA.")
    reference_image = sorted(config.input_dir.glob("*.jpg"))[0]
    print("Draw the ROI: click Bottom-Left, Top-Left, Top-Right, Bottom-Right, then press any key.")
    roi_points = select_roi(str(reference_image))
    run_batch(config, roi_points)


def _cmd_video_to_frames(args: argparse.Namespace) -> None:
    from alina.io_utils import video_to_frames

    video_to_frames(args.video, args.output_dir)


def _cmd_rotate_frames(args: argparse.Namespace) -> None:
    from alina.io_utils import rotate_frames

    rotate_frames(args.input_dir, args.output_dir, angle=args.angle)


def _cmd_resize_images(args: argparse.Namespace) -> None:
    from alina.io_utils import resize_images

    resize_images(args.input_dir, args.output_dir, width=args.width, height=args.height)


def _cmd_cbem(args: argparse.Namespace) -> None:
    create_cbem(args.image, args.output_dir)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    config = EvaluationConfig(canny_dirs=args.canny_dirs, alina_dirs=args.alina_dirs)
    run_evaluation(config)


def _cmd_superimpose(args: argparse.Namespace) -> None:
    overlay_coords(args.image, args.coords, save_to_disk=args.output_dir is not None, output_dir=args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alina")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("label", help="Detect and label taxiway line markings.")
    p.add_argument("--input-dir", dest="input_dir", default=None)
    p.add_argument("--output-images-dir", dest="output_images_dir", required=True)
    p.add_argument("--output-coords-dir", dest="output_coords_dir", required=True)
    p.add_argument("--log-file", dest="log_file", default=None)
    p.add_argument("--peak-threshold", dest="peak_threshold", type=int, default=50)
    p.add_argument("--min-white-pixels", dest="min_white_pixels", type=int, default=200)
    p.add_argument("--circular-threshold", dest="circular_threshold", type=int, default=15)
    p.add_argument("--yellow-lower", dest="yellow_lower", type=int, nargs=3, default=[0, 70, 170], metavar=("H", "S", "V"))
    p.add_argument("--yellow-upper", dest="yellow_upper", type=int, nargs=3, default=[255, 255, 255], metavar=("H", "S", "V"))
    p.add_argument("--mask-ignore-left-columns", dest="mask_ignore_left_columns", type=int, default=300)
    p.add_argument("--image", default=None, help="Process a single image instead of the whole --input-dir.")
    p.add_argument("--show", action="store_true", help="Display the result window (single-image mode only).")
    p.set_defaults(func=_cmd_label)

    p = sub.add_parser("video-to-frames", help="Extract frames from a video file.")
    p.add_argument("--video", required=True)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.set_defaults(func=_cmd_video_to_frames)

    p = sub.add_parser("rotate-frames", help="Rotate a directory of frames.")
    p.add_argument("--input-dir", dest="input_dir", required=True)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.add_argument("--angle", type=int, default=180, choices=[90, 180, 270])
    p.set_defaults(func=_cmd_rotate_frames)

    p = sub.add_parser("resize-images", help="Batch-resize a directory of images.")
    p.add_argument("--input-dir", dest="input_dir", required=True)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.set_defaults(func=_cmd_resize_images)

    p = sub.add_parser("cbem", help="Interactively create a context-based edge map (CBEM) for one frame.")
    p.add_argument("--image", required=True)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.set_defaults(func=_cmd_cbem)

    p = sub.add_parser("evaluate", help="Compute precision/recall of ALINA output vs. CBEM ground truth.")
    p.add_argument("--canny-dirs", dest="canny_dirs", nargs="+", required=True)
    p.add_argument("--alina-dirs", dest="alina_dirs", nargs="+", required=True)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("superimpose", help="Overlay detected coordinates on a Canny edge map.")
    p.add_argument("--image", required=True)
    p.add_argument("--coords", required=True)
    p.add_argument("--output-dir", dest="output_dir", default=None, help="If given, saves the overlay images here.")
    p.set_defaults(func=_cmd_superimpose)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()