from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=10)
    args = parser.parse_args()

    import cv2

    found = []
    for index in range(args.start_index, args.end_index + 1):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"{index}: opened but no frame")
                continue
            h, w = frame.shape[:2]
            print(f"{index}: OK {w}x{h}")
            found.append(index)
        finally:
            cap.release()

    if not found:
        print("No cameras found.")


if __name__ == "__main__":
    main()
