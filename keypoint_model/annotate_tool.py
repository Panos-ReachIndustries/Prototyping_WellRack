"""
Simple OpenCV annotation tool for vial keypoints.

Usage:
    python -m keypoint_model.annotate_tool \
        --images assets/testing_images \
        --output assets/testing_images/vial_keypoints.json

Controls:
    LEFT CLICK   — add keypoint at cursor position
    RIGHT CLICK  — remove nearest keypoint (within 15px)
    'n' / SPACE  — save current & move to next image
    'p'          — go to previous image
    'u'          — undo last added point
    's'          — save annotations to disk
    'q' / ESC    — save & quit
    'r'          — reset all keypoints for current image

Annotations are auto-saved on quit and on each image transition.
"""
import argparse
import json
import os

import cv2
import numpy as np


class VialAnnotator:
    POINT_RADIUS = 4
    POINT_COLOR = (0, 255, 0)
    WINDOW_NAME = "Vial Keypoint Annotator"

    def __init__(self, images_dir: str, output_path: str, max_display: int = 1200):
        self.images_dir = images_dir
        self.output_path = output_path
        self.max_display = max_display

        exts = {".png", ".jpg", ".jpeg"}
        self.file_names = sorted(
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        if not self.file_names:
            raise RuntimeError(f"No images found in {images_dir}")

        if os.path.exists(output_path):
            with open(output_path) as f:
                self.annotations = json.load(f)
            print(f"Loaded existing annotations for {len(self.annotations)} images")
        else:
            self.annotations = {}

        self.current_idx = 0
        self._skip_to_unannotated()
        self.scale = 1.0

    def _skip_to_unannotated(self):
        for i, fname in enumerate(self.file_names):
            if fname not in self.annotations:
                self.current_idx = i
                return

    def _current_file(self):
        return self.file_names[self.current_idx]

    def _get_keypoints(self):
        fname = self._current_file()
        return self.annotations.get(fname, [])

    def _set_keypoints(self, kps):
        self.annotations[self._current_file()] = kps

    def _load_image(self):
        path = os.path.join(self.images_dir, self._current_file())
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        if max(h, w) > self.max_display:
            self.scale = self.max_display / max(h, w)
            img = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
        else:
            self.scale = 1.0
        return img

    def _draw(self, img):
        vis = img.copy()
        kps = self._get_keypoints()
        for i, (x, y) in enumerate(kps):
            dx, dy = int(x * self.scale), int(y * self.scale)
            cv2.circle(vis, (dx, dy), self.POINT_RADIUS, self.POINT_COLOR, -1)
            cv2.putText(vis, str(i), (dx + 6, dy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        status = (f"[{self.current_idx+1}/{len(self.file_names)}] "
                  f"{self._current_file()} | "
                  f"{len(kps)} points | "
                  f"LClick=add RClick=remove n=next p=prev s=save q=quit")
        cv2.putText(vis, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return vis

    def _on_mouse(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox = mx / self.scale
            oy = my / self.scale
            kps = self._get_keypoints()
            kps.append([round(ox, 1), round(oy, 1)])
            self._set_keypoints(kps)

        elif event == cv2.EVENT_RBUTTONDOWN:
            kps = self._get_keypoints()
            if not kps:
                return
            ox = mx / self.scale
            oy = my / self.scale
            pts = np.array(kps, dtype=np.float32)
            dists = np.linalg.norm(pts - np.array([ox, oy]), axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] < 15 / self.scale:
                kps.pop(nearest)
                self._set_keypoints(kps)

    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.annotations, f, indent=2)
        annotated = sum(1 for k in self.annotations if self.annotations[k])
        print(f"Saved {annotated} annotated images → {self.output_path}")

    def run(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

        while True:
            img = self._load_image()
            while True:
                vis = self._draw(img)
                cv2.imshow(self.WINDOW_NAME, vis)
                key = cv2.waitKey(30) & 0xFF

                if key in (ord("q"), 27):
                    self.save()
                    cv2.destroyAllWindows()
                    return

                if key in (ord("n"), ord(" ")):
                    self.current_idx = (self.current_idx + 1) % len(self.file_names)
                    break

                if key == ord("p"):
                    self.current_idx = (self.current_idx - 1) % len(self.file_names)
                    break

                if key == ord("s"):
                    self.save()

                if key == ord("r"):
                    self._set_keypoints([])

                if key == ord("u"):
                    kps = self._get_keypoints()
                    if kps:
                        kps.pop()
                        self._set_keypoints(kps)


def main():
    parser = argparse.ArgumentParser(description="Vial keypoint annotation tool")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="assets/testing_images/vial_keypoints.json")
    args = parser.parse_args()

    annotator = VialAnnotator(args.images, args.output)
    annotator.run()


if __name__ == "__main__":
    main()
