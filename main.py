import cv2
import argparse

from ultralytics import YOLO
import supervision as sv


WANTED_OBJECTS = {0, 26}            # this means 'person' and 'handbag'


def parse_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 live')
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 480],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    frame_count = 0

    args = parse_argparse()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture("rtsp://admin:pass@192.168.0.48:554")
    # cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8s.pt")

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (640, 480))

        if frame_count % 3 == 0:  # Обробляємо кожен n кадр
            result = model(frame)

        cv2.imshow("yolov8", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        frame_count += 1


def main_test():
    frame_count = 0
    args = parse_argparse()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture("rtsp://admin:228a8831Kaf_@192.168.0.48:554")
    # cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8s.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (640, 480))
        if not ret:
            continue
        if frame_count % 3 == 0:
            result = model.predict(frame)
            detections = sv.Detections.from_ultralytics(result[0])
            if detections:
                if set(detections.class_id).intersection(WANTED_OBJECTS):
                    frame = box_annotator.annotate(scene=frame, detections=detections)

        cv2.imshow('test',  cv2.resize(frame, (frame_width, frame_height)))
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        frame_count += 1


if __name__ == '__main__':
    # main()
    main_test()