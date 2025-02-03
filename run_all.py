import subprocess
import argparse

parser = argparse.ArgumentParser(description="Automate license plate detection pipeline.")
parser.add_argument('--video_path', type=str, required=True, help="Path to the video file.")
parser.add_argument('--blur_strength', type=int, default=5, help="Blur strength for license plates.")
args = parser.parse_args()

# Step 1: Run main.py
print("Starting vehicle and license plate detection...")
subprocess.run(['python', 'main.py', '--video_path', args.video_path])

# Step 2: Run add_missing_data.py
print("Interpolating missing data...")
subprocess.run(['python', 'add_missing_data.py'])

# Step 3: Run visualize.py
print("Generating blurred output video...")
subprocess.run(['python', 'visualize.py', '--blur_strength', str(args.blur_strength), '--video_path', args.video_path])

print("Pipeline completed successfully!")
