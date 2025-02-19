import cv2  # isort:skip

import argparse
import os
import subprocess
from functools import partial

import pandas as pd
from imageio_ffmpeg import get_ffmpeg_exe
from pandarallel import pandarallel
from scenedetect import FrameTimecode
from tqdm import tqdm
from moviepy.editor import VideoFileClip, ImageSequenceClip

tqdm.pandas()


def print_log(s, logger=None):
    if logger is not None:
        logger.info(s)
        # pass
    else:
        print(s)


def process_single_row(row, args):
    video_path = row["path"]

    logger = None

    # check mp4 integrity
    # if not is_intact_video(video_path, logger=logger):
    #     return False

    if args.no_scene_split: #如果no_scene_split为True，则按照新的裁剪策略
        split_video_fixed_frames(video_path, args.save_dir, shorter_size=args.shorter_size, output_fps=args.output_fps, logger=None) #ADD output_fps here
        return True


    try:
        if "timestamp" in row:
            timestamp = row["timestamp"]
            if not (timestamp.startswith("[") and timestamp.endswith("]")):
                return False
            scene_list = eval(timestamp)
            scene_list = [(FrameTimecode(s, fps=100), FrameTimecode(t, fps=100)) for s, t in scene_list]
        else:
            scene_list = [None]
        if args.drop_invalid_timestamps:
            return True
    except Exception as e:
        if args.drop_invalid_timestamps:
            return False

    if "relpath" in row:
        save_dir = os.path.dirname(os.path.join(args.save_dir, row["relpath"]))
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = args.save_dir

    shorter_size = args.shorter_size
    if (shorter_size is not None) and ("height" in row) and ("width" in row):
        min_size = min(row["height"], row["width"])
        if min_size <= shorter_size:
            shorter_size = None

    split_video_scene_detect(
        video_path,
        scene_list,
        save_dir=save_dir,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        target_fps=args.target_fps,
        shorter_size=shorter_size,
        logger=logger,
    )
    return True

def split_video_scene_detect(
    video_path,
    scene_list,
    save_dir,
    min_seconds=2,
    max_seconds=15,
    target_fps=30,
    shorter_size=None,
    verbose=False,
    logger=None,
):
    """
    scenes shorter than min_seconds will be ignored;
    scenes longer than max_seconds will be cut to save the beginning max_seconds.
    Currently, the saved file name pattern is f'{fname}_scene-{idx}'.mp4

    Args:
        scene_list (List[Tuple[FrameTimecode, FrameTimecode]]): each element is (s, t): start and end of a scene.
        min_seconds (float | None)
        max_seconds (float | None)
        target_fps (int | None)
        shorter_size (int | None)
    """
    FFMPEG_PATH = get_ffmpeg_exe()

    save_path_list = []
    for idx, scene in enumerate(scene_list):
        if scene is not None:
            s, t = scene  # FrameTimecode
            if min_seconds is not None:
                if (t - s).get_seconds() < min_seconds:
                    continue

            duration = t - s
            if max_seconds is not None:
                fps = s.framerate
                max_duration = FrameTimecode(max_seconds, fps=fps)
                duration = min(max_duration, duration)

        # save path
        fname = os.path.basename(video_path)
        fname_wo_ext = os.path.splitext(fname)[0]
        # TODO: fname pattern
        save_path = os.path.join(save_dir, f"{fname_wo_ext}_scene-{idx}.mp4")
        if os.path.exists(save_path):
            # print_log(f"File '{save_path}' already exists. Skip.", logger=logger)
            continue
        
        # ffmpeg cmd
        cmd = [FFMPEG_PATH]

        # Only show ffmpeg output for the first call, which will display any
        # errors if it fails, and then break the loop. We only show error messages
        # for the remaining calls.
        # cmd += ['-v', 'error']

        # clip to cut
        # Note: -ss after -i is very slow; put -ss before -i !!!
        if scene is None:
            cmd += ["-nostdin", "-y", "-i", video_path]
        else:
            cmd += ["-nostdin", "-y", "-ss", str(s.get_seconds()), "-i", video_path, "-t", str(duration.get_seconds())]

        # target fps
        if target_fps is not None:
            cmd += ["-r", f"{target_fps}"]

        # aspect ratio
        if shorter_size is not None:
            cmd += ["-vf", f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'"]
            # cmd += ['-vf', f"scale='if(gt(iw,ih),{shorter_size},trunc(ow/a/2)*2)':-2"]

        cmd += ["-map", "0:v", save_path]
        # print(cmd)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = proc.communicate()
        # stdout = stdout.decode("utf-8")
        # print_log(stdout, logger=logger)

        save_path_list.append(video_path)
        if verbose:
            print_log(f"Video clip saved to '{save_path}'", logger=logger)

    return save_path_list


def convert_video_to_cfr(input_path, output_path, target_fps, logger=None):
    try:
        clip = VideoFileClip(input_path)
        clip = clip.set_fps(target_fps)
        clip.write_videofile(output_path, fps=target_fps, codec="libx264", audio_codec="aac", logger=None) # 可以设置更多参数，如编码器
        print_log(f"Converted video to fixed FPS: {output_path}", logger=logger)
        clip.close()
        return True
    except Exception as e:
        print_log(f"Error converting video: {e}", logger=logger)
        return False


def split_video_fixed_frames(video_path, save_dir, shorter_size=None, logger=None, output_fps: int = 12):
    fname = os.path.basename(video_path)
    fname_wo_ext = os.path.splitext(fname)[0]
    temp_video_path = os.path.join(save_dir, f"{fname_wo_ext}_temp.mp4") # 临时文件路径

    # 转换为固定帧率
    if not convert_video_to_cfr(video_path, temp_video_path, output_fps, logger):
        print_log(f"Failed to convert {video_path} to fixed FPS. Skipping.", logger=logger)
        return

    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < 17:
        print_log(f"Video '{video_path}' has fewer than 17 frames after conversion. Skipping.", logger=logger)
        cap.release()
        os.remove(temp_video_path)  # 删除临时文件
        return

    start_frames = [0, total_frames // 2, max(0, total_frames - 17)]

    for i, start_frame in enumerate(start_frames):
        if start_frame + 17 > total_frames:
            print_log(f"Not enough frames for clip {i} in '{video_path}'. Skipping.", logger=logger)
            continue

        save_path = os.path.join(save_dir, f"{fname_wo_ext}_{i}.mp4")
        if os.path.exists(save_path):
            print_log(f"File '{save_path}' already exists.  Skipping.", logger=logger)
            continue

        # 读取帧 using OpenCV
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(17):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        if len(frames) != 17:
            continue

        # 转换帧为 MoviePy clip
        frame_duration = 1 / output_fps
        original_duration = [frame_duration] * len(frames)
        clip = ImageSequenceClip(frames, durations=original_duration)


        # 调整大小 if shorter_size is specified
        if shorter_size is not None:
            original_width, original_height = clip.size
            if original_width > original_height:
                new_width = int(shorter_size * original_width / original_height)
                clip = clip.resize((new_width, shorter_size))
            else:
                new_height = int(shorter_size * original_height / original_width)
                clip = clip.resize((shorter_size, new_height))

        clip.write_videofile(save_path, fps=output_fps, logger=None)

        print_log(f"Video clip saved to '{save_path}'", logger=logger)

    cap.release()
    os.remove(temp_video_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument(
        "--min_seconds", type=float, default=None, help="if not None, clip shorter than min_seconds is ignored"
    )
    parser.add_argument(
        "--max_seconds", type=float, default=None, help="if not None, clip longer than max_seconds is truncated"
    )
    parser.add_argument("--target_fps", type=int, default=None, help="target fps of clips")
    parser.add_argument(
        "--shorter_size", type=int, default=None, help="resize the shorter size by keeping ratio; will not do upscale"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for pandarallel")
    parser.add_argument("--disable_parallel", action="store_true", help="disable parallel processing")
    parser.add_argument("--drop_invalid_timestamps", action="store_true", help="drop rows with invalid timestamps")
    parser.add_argument("--no_scene_split",action="store_true",help="If true, don't use scene detect split but takes the start, mid, end-16 frames.")
    parser.add_argument("--output_fps", type=int, default=None, help="Set a fixed frame rate for the output video")  # NEW ARGUMENT
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    # create save_dir
    os.makedirs(args.save_dir, exist_ok=True)

    # initialize pandarallel
    if not args.disable_parallel:
        if args.num_workers is not None:
            pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
        else:
            pandarallel.initialize(progress_bar=True)
    process_single_row_partial = partial(process_single_row, args=args)

    # process
    meta = pd.read_csv(args.meta_path)
    if not args.disable_parallel:
        results = meta.parallel_apply(process_single_row_partial, axis=1)
    else:
        results = meta.apply(process_single_row_partial, axis=1)
    if args.drop_invalid_timestamps:
        meta = meta[results]
        assert args.meta_path.endswith("timestamp.csv"), "Only support *timestamp.csv"
        meta.to_csv(args.meta_path.replace("timestamp.csv", "correct_timestamp.csv"), index=False)
        print(f"Corrected timestamp file saved to '{args.meta_path.replace('timestamp.csv', 'correct_timestamp.csv')}'")
if __name__ == "__main__":
    main()