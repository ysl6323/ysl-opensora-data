import cv2
import os

def process_video(input_path, output_prefix):
    # 读取视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = 15
    interval = 1.0 / target_fps  # 目标帧间隔

    sampled_frames = []  # 存储采样后的所有帧
    current_target_time = 0.0  # 当前目标时间点
    frame_count = 0  # 已处理的帧计数器

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束或读取失败

        current_time = frame_count / original_fps  # 计算当前帧时间
        frame_count += 1

        # 当当前时间超过或等于目标时间时，采样该帧
        if current_time >= current_target_time:
            sampled_frames.append(frame.copy())  # 复制帧以避免覆盖
            current_target_time += interval  # 更新下一个目标时间点

    cap.release()

    total_sampled = len(sampled_frames)
    if total_sampled == 0:
        print("未采样到任何帧")
        return

    # 提取首、中、尾各17帧
    def get_segment(frames, start, length=17):
        end = start + length
        return frames[start:end] if end <= len(frames) else frames[start:]

    start_segment = get_segment(sampled_frames, 0)
    mid_start = max(0, (total_sampled - 17) // 2)
    mid_segment = get_segment(sampled_frames, mid_start)
    end_start = max(0, total_sampled - 17)
    end_segment = get_segment(sampled_frames, end_start)

    # 保存视频函数
    def save_segment(segment, suffix):
        if len(segment) == 0:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f"{output_prefix}_{suffix}.mp4"
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        for frame in segment:
            out.write(frame)
        out.release()
        print(f"已保存: {os.path.abspath(output_path)}")

    # 保存三个视频段
    save_segment(start_segment, "start")
    save_segment(mid_segment, "middle")
    save_segment(end_segment, "end")

# 示例使用
if __name__ == "__main__":
    input_video = "input.mp4"  # 输入视频路径
    output_prefix = "output"   # 输出视频前缀
    process_video(input_video, output_prefix)