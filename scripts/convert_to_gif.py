#!/usr/bin/env python3
"""
Convert recorded MP4 evaluation videos to lightweight animated GIFs for README showcase.
"""

import argparse
import os
import sys
import imageio


def convert_mp4_to_gif(mp4_path: str, gif_path: str, fps: int = 15, max_duration: float = 12.0) -> None:
    if not os.path.exists(mp4_path):
        print(f"[!] Input video not found: {mp4_path}")
        return

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    print(f"[*] Reading '{mp4_path}'...")
    reader = imageio.get_reader(mp4_path)
    meta = reader.get_meta_data()
    orig_fps = meta.get("fps", 30)

    # Frame skipping to target fps
    step = max(1, int(round(orig_fps / fps)))
    frames = []

    total_frames = int(orig_fps * max_duration)
    for i, frame in enumerate(reader):
        if i >= total_frames:
            break
        if i % step == 0:
            frames.append(frame)

    reader.close()

    if not frames:
        print("[!] No frames extracted from video.")
        return

    print(f"[*] Writing {len(frames)} frames to '{gif_path}' at {fps} fps...")
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    file_size_kb = os.path.getsize(gif_path) / 1024
    print(f"[+] Successfully generated '{gif_path}' ({file_size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Convert MP4 to optimized GIF")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input MP4")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output GIF")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS (default: 15)")
    parser.add_argument("--max-duration", type=float, default=12.0, help="Max duration in seconds (default: 12.0)")
    args = parser.parse_args()

    convert_mp4_to_gif(args.input, args.output, fps=args.fps, max_duration=args.max_duration)


if __name__ == "__main__":
    main()
