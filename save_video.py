import argparse
from evaluate import evaluate

# 화면 없이 주행 영상(GIF)만 저장
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 에이전트의 주행 영상을 GIF로 저장")
    parser.add_argument("--ckpt", default="latest", help="반복 횟수, 'latest', 'best'")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--video", default="videos/eval_run.gif")
    args = parser.parse_args()
    evaluate(args.ckpt, args.output_dir, render=False, video_path=args.video)
