import os
import pandas as pd
import av
import torch
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

def read_video_pyav(container, indices):
    '''
    使用 PyAV 解码视频。
    参数:
        container (`av.container.input.InputContainer`): PyAV 容器。
        indices (`List[int]`): 要解码的帧的索引列表。
    返回:
        result (np.ndarray): 解码后的帧数组，形状为 (num_frames, height, width, 3)。
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def process_video(video_path, model, processor, num_frames_sampled=8):
    '''
    处理单个视频并生成模型的描述。
    参数:
        video_path (str): 视频文件的路径。
        model: 预训练的 LlavaNextVideoForConditionalGeneration 模型。
        processor: 预训练的 LlavaNextVideoProcessor 处理器。
        num_frames_sampled (int): 从视频中抽取的帧数。
    返回:
        response (str): 模型生成的视频描述。
    '''
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / num_frames_sampled).astype(int)
    video = read_video_pyav(container, indices)
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences."},
                {"type": "video"},
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=video, return_tensors="pt").to(device)
    
    out = model.generate(**inputs, max_new_tokens=300)
    response = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return response[0].split("ASSISTANT:")[1].strip()

def process_videos_in_range(df, start_idx=None, end_idx=None):
    '''
    处理指定范围的视频行，默认处理所有行。
    参数:
        df (pd.DataFrame): 输入的 DataFrame，包含视频路径和其他信息。
        start_idx (int, optional): 开始处理的行索引。
        end_idx (int, optional): 结束处理的行索引。
    返回:
        output_df (pd.DataFrame): 包含视频描述的 DataFrame。
    '''
    output_df = pd.DataFrame(columns=df.columns)
    
    # 如果 start_idx 和 end_idx 为 None，处理所有行
    if start_idx is None and end_idx is None:
        rows_to_process = df.iterrows()
    else:
        rows_to_process = df.iloc[start_idx:end_idx].iterrows()

    # 遍历并处理指定的行
    for index, row in rows_to_process:
        video_path = row['video_path']
        
        if os.path.exists(video_path):
            print(f"正在处理视频：{video_path}")
            description = process_video(video_path, model, processor)
        else:
            description = "视频路径无效"
        
        # 替换原来的 description（text），保留其他列
        new_row = row.copy()
        new_row['text'] = description
        
        # 更新输出 DataFrame
        # output_data.append(new_row)
        output_df = output_df.append(new_row, ignore_index=True)
        
        # 每处理10个视频就保存一次
        if (index + 1) % 10 == 0:
            output_df.to_csv(output_csv_path, index=False)
            print(f"已保存前 {index + 1} 个视频的描述。")

    # 处理完所有视频后，保存最终结果
    if output_df:
        output_df.to_csv(output_csv_path, index=False)

    print("所有视频的描述已处理并保存完毕。")


# 加载模型（半精度）
model = LlavaNextVideoForConditionalGeneration.from_pretrained("/data/baotang/sana_video/generate/model/LLaVA-NeXT-Video-7B", torch_dtype=torch.float16, device_map=device)
processor = LlavaNextVideoProcessor.from_pretrained("/data/baotang/sana_video/generate/model/LLaVA-NeXT-Video-7B")

# 读取原始 CSV 文件
input_csv_path = '/data/baotang/sana_video/ucf101.csv'  # 替换为原 CSV 文件的路径
df = pd.read_csv(input_csv_path)

# 如果 final_output.csv 已经存在，读取它，否则创建一个新的 CSV 文件
output_csv_path = '/data/baotang/sana_video/generate/ucf4_output.csv'
if os.path.exists(output_csv_path):
    raise AssertionError("Output file already exists")

# 控制要处理的行范围，传入 None 时处理所有行
# row_range = None  # 可以修改为 [start_idx, end_idx] 来控制处理的行范围
# row_range = [0, 4000]
# row_range = [4001, 8000]
# row_range = [8001, 12000]
row_range = [12001, 16000]

if row_range is None:
    process_videos_in_range(df)
else:
    start_idx, end_idx = row_range
    process_videos_in_range(df, start_idx, end_idx)