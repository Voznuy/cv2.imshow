import cv2
import argparse

from ultralytics import YOLO
import supervision as sv


class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
            ]


def parse_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 live')
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_argparse()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture("rtsp://admin:228a8831Kaf_@192.168.0.48:554")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    while True:
        ret, frame = cap.read()
        results = model(frame)[0]

        boxes = results.boxes.xyxy.cpu().numpy()

        class_ids = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()

        detections = sv.Detections(
            xyxy=boxes,
            class_id=class_ids.astype(int),
            confidence=confidences
        )

        labels = [class_names[int(class_id)] for class_id in detections.class_id.astype(int)]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow('test', cv2.resize(frame, (frame_width, frame_height)))
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    # main()
    main()
