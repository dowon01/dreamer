import os
import re

def checkpoint_paths(output_dir, tag):
    # tag: 반복 횟수(int) 또는 "best" 같은 이름
    suffix = f"iter_{tag}" if isinstance(tag, int) else tag
    return {
        'wm': os.path.join(output_dir, f"wm_{suffix}.pth"),
        'actor': os.path.join(output_dir, f"actor_{suffix}.pth"),
        'critic': os.path.join(output_dir, f"critic_{suffix}.pth"),
        'train_state': os.path.join(output_dir, f"train_state_{suffix}.pth"),
    }

def find_latest_iter(output_dir):
    if not os.path.isdir(output_dir):
        return None
    iters = [int(m.group(1)) for f in os.listdir(output_dir)
             if (m := re.fullmatch(r"wm_iter_(\d+)\.pth", f))]
    return max(iters) if iters else None

def resolve_tag(output_dir, ckpt):
    # "latest" | "best" | 숫자 문자열 -> checkpoint_paths에 넘길 tag
    if ckpt == "latest":
        latest = find_latest_iter(output_dir)
        if latest is None:
            raise FileNotFoundError(f"{output_dir}에 체크포인트가 없습니다")
        return latest
    return int(ckpt) if ckpt.isdigit() else ckpt
