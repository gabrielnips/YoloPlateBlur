import ast
import cv2
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description="Visualize video with blurred license plates.")
parser.add_argument('--blur_strength', type=int, default=5, help="Blur strength (kernel size). Default value: 5")
parser.add_argument('--video_path', type=str, required=True, help="Path to the video file. Default value: './input/sample.mp4'")
args = parser.parse_args()

results = pd.read_csv('./test_interpolated.csv')


video_path = args.video_path
cap = cv2.VideoCapture(video_path)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./output/out.mp4', fourcc, fps, (width, height))

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            license_plate_region = frame[int(y1):int(y2), int(x1):int(x2)]  
            blurred_region = cv2.GaussianBlur(license_plate_region, (99, 99), args.blur_strength)  
            frame[int(y1):int(y2), int(x1):int(x2)] = blurred_region 

        out.write(frame) 

out.release()
cap.release()