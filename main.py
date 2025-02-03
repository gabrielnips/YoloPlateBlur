from ultralytics import YOLO
import cv2
import numpy as np
import argparse
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import time
import sys

# Barra de progresso animada
def animated_progress():
    animations = ['[.]', '[/]', '[|]', '[\]',]
    while True:
        for anim in animations:
            yield anim

progress = animated_progress()

results = {}
mot_tracker = Sort()

# Carregar modelos com logs desativados
coco_model = YOLO('yolov8n.pt')
coco_model.overrides['verbose'] = False

license_plate_detector = YOLO('license_plate_detector.pt')
license_plate_detector.overrides['verbose'] = False

# Argumentos
parser = argparse.ArgumentParser(description="Detect and read license plates from a video.")
parser.add_argument('--video_path', type=str, required=True, help="Path to the video file. Default value: './input/sample.mp4'")
args = parser.parse_args()

# Carregar vídeo
cap = cv2.VideoCapture(args.video_path)
vehicles = [2, 3, 5, 7]

# Processar frames
frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Detecção de veículos
        detections = coco_model(frame, verbose=False)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Rastreamento de veículos
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detecção de placas
        license_plates = license_plate_detector(frame, verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Associar placa ao carro
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

        # Atualizar barra de progresso
        sys.stdout.write(f'\rProcessing Frames {next(progress)}')
        sys.stdout.flush()

cap.release()
print("\nProcessing complete.")

# Salvar resultados
write_csv(results, './test.csv')
