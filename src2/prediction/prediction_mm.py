import argparse
import os
from prediction_model import process
import mmengine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--out', default="mm_result")
    parser.add_argument('--config', default="config/prediction_slowfast.py")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(config["cfg_options"])

    video_paths = [os.path.join(args.dir, f) for f in os.listdir(args.dir)]
    print(video_paths)
    for video in video_paths:
        video_base_name = video.split("/")[-1]
        out = os.path.join(args.out, video_base_name)
        print(out)
        os.makedirs(out, exist_ok=True)
        process(video, out, config)

if __name__ == "__main__":
    main()

