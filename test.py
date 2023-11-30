from ultralytics import YOLO
import cv2
import math


cap = cv2.VideoCapture("rtsp://admin:228a8831Kaf_@192.168.0.48:554")

model = YOLO("yolov8s.pt")

classNames = ['person']

