#!/usr/bin/env python3
"""
Realtime person detection using YOLOv8m and OpenCV.
- Captures frames from the default camera (or a provided source)
- Detects people (class "person") in each frame
- Draws green rectangles around every detected person
- Logs timestamp and number of people detected to a text file each loop iteration
- Attempts to run the loop at a 3 ms cadence (configurable)

Notes:
- Actual loop rate will be bounded by inference and display time.
- Press 'q' to quit the display window.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, List

import cv2  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime person detection with YOLOv8m and OpenCV")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (e.g., '0') or path to video file",
    )
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=3.0,
        help="Target loop interval in milliseconds (default: 3 ms)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for person detections (default: 0.25)",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="yolov8m.onnx",
        help="Path to YOLOv8m ONNX model (used if Ultralytics backend unavailable)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="people_log.txt",
        help="Path to the log file where timestamp and count are written",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window (useful for headless environments)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional capture width to request from camera",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional capture height to request from camera",
    )
    return parser.parse_args()


def ensure_onnx_model(onnx_path: str) -> str:
    """Ensure the ONNX model exists locally; download yolov8m.onnx if missing.

    Returns the path to the ONNX file.
    """
    if os.path.exists(onnx_path) and os.path.isfile(onnx_path):
        return onnx_path
    # Attempt to download from Ultralytics release bucket
    default_url = (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.onnx"
    )
    try:
        import urllib.request

        print(f"[INFO] Downloading ONNX model to '{onnx_path}' ...")
        urllib.request.urlretrieve(default_url, onnx_path)
        print("[INFO] Download complete.")
        return onnx_path
    except Exception as ex:
        raise RuntimeError(
            f"Could not download ONNX model to '{onnx_path}'. Provide it manually. Error: {ex}"
        )


def letterbox_image(image, new_shape: Tuple[int, int] = (640, 640)):
    """Resize image with unchanged aspect ratio using padding (like YOLO letterbox)."""
    height, width = image.shape[:2]
    r = min(new_shape[0] / height, new_shape[1] / width)
    new_unpad = (int(round(width * r)), int(round(height * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (width, height) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    color = (114, 114, 114)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def infer_persons_opencv_dnn(
    net,
    frame,
    input_size: Tuple[int, int] = (640, 640),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """Run YOLOv8 ONNX with OpenCV DNN and return person boxes in xyxy.

    This function assumes the ONNX output is in the standard Ultralytics format:
    [num_detections, 84] with [x,y,w,h, conf, 80 class probs].
    """
    img, ratio, (dw, dh) = letterbox_image(frame, input_size)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=input_size, swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()  # shape: (1, N, 84)

    if preds.ndim == 3:
        preds = preds[0]
    elif preds.ndim == 2:
        preds = preds
    else:
        preds = preds.reshape((-1, preds.shape[-1]))

    boxes: List[Tuple[int, int, int, int]] = []
    confidences: List[float] = []

    frame_h, frame_w = frame.shape[:2]

    for det in preds:
        x, y, w, h = det[0:4]
        obj_conf = det[4]
        class_scores = det[5:]
        class_id = int(class_scores.argmax())
        class_conf = float(class_scores[class_id])
        score = float(obj_conf) * class_conf
        if class_id != 0:
            continue  # only persons
        if score < conf_threshold:
            continue

        # Convert from center x,y,w,h to top-left x1,y1,x2,y2 in the letterboxed image
        cx, cy, bw, bh = float(x), float(y), float(w), float(h)
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Undo letterbox transform
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = max(0, min(frame_w - 1, int(round(x1))))
        y1 = max(0, min(frame_h - 1, int(round(y1))))
        x2 = max(0, min(frame_w - 1, int(round(x2))))
        y2 = max(0, min(frame_h - 1, int(round(y2))))

        boxes.append((x1, y1, x2, y2))
        confidences.append(score)

    # Basic NMS to reduce overlaps
    if boxes:
        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=[(b[0], b[1], b[2]-b[0], b[3]-b[1]) for b in boxes],
            scores=confidences,
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold,
        )
        if len(nms_indices) > 0:
            nms_indices = nms_indices.flatten().tolist()
            boxes = [boxes[i] for i in nms_indices]
            confidences = [confidences[i] for i in nms_indices]

    return boxes, confidences


def resolve_source(source_str: str):
    """Return int camera index if numeric, else the string path as-is."""
    try:
        # Accept values like "0", "1" as camera indices
        return int(source_str)
    except ValueError:
        return source_str


def open_video_capture(source, width: Optional[int], height: Optional[int]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


def draw_green_rectangle(frame, xyxy: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def format_timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    args = parse_args()

    source = resolve_source(args.source)
    loop_interval_seconds: float = max(0.0, args.interval_ms / 1000.0)
    display_enabled: bool = not args.no_display

    # Try Ultralytics first; if not available, fall back to OpenCV DNN + ONNX
    backend: str = "ultralytics"
    yolo_model = None
    try:
        from ultralytics import YOLO  # type: ignore

        try:
            yolo_model = YOLO("yolov8m.pt")
        except Exception as ex:
            print(f"[WARN] Ultralytics model load failed: {ex}", file=sys.stderr)
            backend = "opencv_dnn"
    except Exception as ex:
        print(f"[WARN] Ultralytics not available: {ex}", file=sys.stderr)
        backend = "opencv_dnn"

    # Prepare video capture
    cap = open_video_capture(source, args.width, args.height)
    if not cap.isOpened():
        print("[ERROR] Unable to open video source.", file=sys.stderr)
        return 2

    # Prepare logging (append mode)
    log_path = args.log_file
    os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
    try:
        log_file = open(log_path, mode="a", encoding="utf-8")
    except Exception as ex:
        print(f"[ERROR] Failed to open log file '{log_path}': {ex}", file=sys.stderr)
        cap.release()
        return 3

    window_name = "Person Detection (YOLOv8m)"
    if display_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        # Prepare OpenCV DNN if Ultralytics is not available
        net = None
        input_size: Tuple[int, int] = (640, 640)
        if backend == "opencv_dnn":
            try:
                onnx_path = ensure_onnx_model(args.onnx_model)
                net = cv2.dnn.readNetFromONNX(onnx_path)
                # Prefer CUDA if available in the environment; otherwise CPU
                try:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                except Exception:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as ex:
                print(f"[ERROR] Failed to initialize OpenCV DNN with ONNX: {ex}", file=sys.stderr)
                return 1

        while True:
            loop_start = time.perf_counter()

            success, frame = cap.read()
            if not success or frame is None:
                print("[WARN] Failed to read frame. Exiting loop.", file=sys.stderr)
                break

            person_count = 0

            if backend == "ultralytics":
                # Run Ultralytics inference
                try:
                    results = yolo_model(
                        frame,
                        verbose=False,
                        conf=float(args.confidence),
                        classes=[0],  # person
                    )
                except Exception as ex:
                    print(f"[ERROR] Inference failed: {ex}", file=sys.stderr)
                    break

                # Iterate over results (usually a single result for one frame)
                for result in results:
                    boxes = getattr(result, "boxes", None)
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):
                        try:
                            cls_id = int(boxes.cls[i].item())  # type: ignore[attr-defined]
                            conf = float(boxes.conf[i].item())  # type: ignore[attr-defined]
                            if cls_id != 0 or conf < args.confidence:
                                continue
                            xyxy_tensor = boxes.xyxy[i]  # type: ignore[attr-defined]
                            x1, y1, x2, y2 = [int(v) for v in xyxy_tensor.tolist()]
                            person_count += 1
                            draw_green_rectangle(frame, (x1, y1, x2, y2))
                        except Exception:
                            continue
            else:
                # OpenCV DNN path (ONNX)
                try:
                    boxes_xyxy, confidences = infer_persons_opencv_dnn(
                        net=net,
                        frame=frame,
                        input_size=input_size,
                        conf_threshold=float(args.confidence),
                    )
                except Exception as ex:
                    print(f"[ERROR] DNN inference failed: {ex}", file=sys.stderr)
                    break

                for (x1, y1, x2, y2) in boxes_xyxy:
                    draw_green_rectangle(frame, (x1, y1, x2, y2))
                person_count = len(boxes_xyxy)

            # Log timestamp and person count
            timestamp_iso = format_timestamp_utc()
            try:
                log_file.write(f"{timestamp_iso},{person_count}\n")
                log_file.flush()
            except Exception as ex:
                print(f"[WARN] Failed to write to log file: {ex}", file=sys.stderr)

            # Display annotated frame if enabled
            if display_enabled:
                cv2.imshow(window_name, frame)
                # Use a very small wait to keep the window responsive
                # Note: waitKey returns -1 if no key; otherwise lower 8-bits is key code
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Enforce loop cadence (best-effort)
            if loop_interval_seconds > 0:
                elapsed = time.perf_counter() - loop_start
                remaining = loop_interval_seconds - elapsed
                if remaining > 0:
                    # sleep granularity is OS-dependent; sub-10ms sleeps may be imprecise
                    time.sleep(remaining)

    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        pass
    finally:
        try:
            log_file.close()
        except Exception:
            pass
        cap.release()
        if display_enabled:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
