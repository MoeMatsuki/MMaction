import os
import os.path as osp
import cv2
import argparse

def frame_extraction(video_path):
    """Extract frames given video_path.
    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    print(f"target video is {video_path}")
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    args = parser.parse_args()
    return args

args = parse_args()
frame_paths, original_frames = frame_extraction(args.video)