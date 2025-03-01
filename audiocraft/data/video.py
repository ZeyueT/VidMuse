import decord
from decord import VideoReader
from decord import cpu
import torch
import math
import einops
import torchvision.transforms as transforms

def adjust_video_duration(video_tensor, duration, target_fps):
    current_duration = video_tensor.shape[1]
    target_duration = duration * target_fps

    if current_duration > target_duration:
        video_tensor = video_tensor[:, :target_duration]
    elif current_duration < target_duration:
        last_frame = video_tensor[:, -1:]
        repeat_times = target_duration - current_duration
        video_tensor = torch.cat((video_tensor, last_frame.repeat(1, repeat_times, 1, 1)), dim=1)

    return video_tensor

def video_read_local(filepath, seek_time=0., duration=-1, target_fps=2):
    vr = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps()

    if duration > 0:
        total_frames_to_read = target_fps * duration
        frame_interval = int(math.ceil(fps / target_fps))
        start_frame = int(seek_time * fps)
        end_frame = start_frame + frame_interval * total_frames_to_read
        frame_ids = list(range(start_frame, min(end_frame, len(vr)), frame_interval))
    else:
        frame_ids = list(range(0, len(vr), int(math.ceil(fps / target_fps))))

    frames = vr.get_batch(frame_ids)
    frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
    
    resize_transform = transforms.Resize((224, 224))
    frames = [resize_transform(frame) for frame in frames]
    video_tensor = torch.stack(frames)
    video_tensor = einops.rearrange(video_tensor, 't c h w -> c t h w') # [T, C, H, W] -> [C, T, H, W]
    video_tensor = adjust_video_duration(video_tensor, duration, target_fps)
    assert video_tensor.shape[1] == duration * target_fps, f"the shape of video_tensor is {video_tensor.shape}"

    return video_tensor


def video_read_global(filepath, seek_time=0., duration=-1, target_fps=2, global_mode='average', global_num_frames=32):
    vr = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frame_count = len(vr)

    if duration > 0:
        total_frames_to_read = target_fps * duration
        frame_interval = int(math.ceil(fps / target_fps))
        start_frame = int(seek_time * fps)
        end_frame = start_frame + frame_interval * total_frames_to_read
        frame_ids = list(range(start_frame, min(end_frame, frame_count), frame_interval))
    else:
        frame_ids = list(range(0, frame_count, int(math.ceil(fps / target_fps))))

    local_frames = vr.get_batch(frame_ids)
    local_frames = torch.from_numpy(local_frames.asnumpy()).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
    
    resize_transform = transforms.Resize((224, 224))
    local_frames = [resize_transform(frame) for frame in local_frames]
    local_video_tensor = torch.stack(local_frames)
    local_video_tensor = einops.rearrange(local_video_tensor, 't c h w -> c t h w') # [T, C, H, W] -> [C, T, H, W]
    local_video_tensor = adjust_video_duration(local_video_tensor, duration, target_fps)

    if global_mode=='average':
        global_frame_ids = torch.linspace(0, frame_count - 1, global_num_frames).long()

        global_frames = vr.get_batch(global_frame_ids)
        global_frames = torch.from_numpy(global_frames.asnumpy()).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        
        global_frames = [resize_transform(frame) for frame in global_frames]
        global_video_tensor = torch.stack(global_frames)
        global_video_tensor = einops.rearrange(global_video_tensor, 't c h w -> c t h w') # [T, C, H, W] -> [C, T, H, W]

    assert global_video_tensor.shape[1] == global_num_frames, f"the shape of global_video_tensor is {global_video_tensor.shape}"
    return local_video_tensor, global_video_tensor

