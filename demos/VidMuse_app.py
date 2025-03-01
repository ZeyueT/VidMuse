import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
import subprocess as sp
import sys
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings

from einops import rearrange
import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import VidMuse, MultiBandDiffusion

from moviepy.editor import VideoFileClip
import decord
from decord import VideoReader
from decord import cpu
import math
import einops
import torchvision.transforms as transforms


MODEL = None
SPACE_ID = os.environ.get('SPACE_ID', '')

INTERRUPTING = False

_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomiting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)

sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True

class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break
                
file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='facebook/musicgen-melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        # Clear PyTorch CUDA cache and delete model
        del MODEL
        torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = VidMuse.get_pretrained(version)


def get_video_duration(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration_sec = clip.duration
        clip.close()
        return duration_sec
    except Exception as e:
        print(f"Error: {e}")
        return None


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

def merge_audio_video(video_path, audio_path, output_path):

    command = [
        'ffmpeg',
        '-i', video_path,        
        '-i', audio_path,        
        '-c:v', 'copy',          
        '-c:a', 'aac',           
        '-map', '0:v:0',         
        '-map', '1:a:0',         
        '-shortest',             
        '-strict', 'experimental',
        output_path              
    ]
    
    try:
        sp.run(command, check=True)
        print(f"Successfully merged audio and video into {output_path}")
        return output_path
    except sp.CalledProcessError as e:
        print(f"Error merging audio and video: {e}")
        return None


def _do_predictions(video, duration, progress=False, gradio_progress=None, **gen_kwargs):
    
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    fps = 2
    progress=True
    USE_DIFFUSION=False

    video_path = video[0]
    duration = int(get_video_duration(video_path))
    MODEL.set_generation_params(duration=duration, **gen_kwargs)

    local_video_tensor, global_video_tensor = video_read_global(video_path, seek_time=0., duration=duration, target_fps=fps)

    try:
        outputs = MODEL.generate([local_video_tensor, global_video_tensor], progress=progress, return_tokens=USE_DIFFUSION)

    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])

    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)


            merged_video_path = f"./demo_result/merged_video_{os.path.basename(file.name)[:-4]}.mp4"
            directory = os.path.dirname(merged_video_path)
            if not os.path.exists(directory):
                os.makedirs(directory)            

            pending_videos.append(
                pool.submit(merge_audio_video, video_path, file.name, merged_video_path)
            )

            out_wavs.append(file.name)
            file_cleaner.add(file.name)

    out_videos = [merged_video_path]
    for video in out_videos:
        file_cleaner.add(video)
    print("batch finished", time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))

    time.sleep(8)
    return out_videos, out_wavs




js = """
function createWaveAnimation() {
    const text = document.getElementById('text');
    var i = 0;
    setInterval(function() {
        const colors = [
            'red, orange, yellow, green, blue, indigo, violet, purple',
            'orange, yellow, green, blue, indigo, violet, purple, red',
            'yellow, green, blue, indigo, violet, purple, red, orange',
            'green, blue, indigo, violet, purple, red, orange, yellow',
            'blue, indigo, violet, purple, red, orange, yellow, green',
            'indigo, violet, purple, red, orange, yellow, green, blue',
            'violet, purple, red, orange, yellow, green, blue, indigo',
            'purple, red, orange, yellow, green, blue, indigo, violet',
        ];
        const angle = 45;
        const colorIndex = i % colors.length;
        text.style.background = `linear-gradient(${angle}deg, ${colors[colorIndex]})`;
        text.style.webkitBackgroundClip = 'text';
        text.style.backgroundClip = 'text';
        text.style.color = 'transparent';
        text.style.fontSize = '28px';
        text.style.width = 'auto';
        text.textContent = 'VidMuse';
        text.style.fontWeight = 'bold';
        i += 1;
    }, 200);
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    // console.log(url_params);
    // console.log('hello world...');
    // console.log(window.location.search);
    // console.log('hello world...');
    // alert(window.location.search)
    // alert(url_params);
    return url_params;
}

"""
title_html= """
<h2> <span class="gradient-text" id="text">🎶 VidMuse</span><span class="plain-text">: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling</span></h2>
<a href="https://vidmuse.github.io/">[[Project Page 🎥]</a> 
<a href="https://arxiv.org/abs/2406.04321/">[Paper 🗞️]</a> 
<a href="https://github.com/ZeyueT/VidMuse/">[Code 🔎]</a> 
<a href="https://huggingface.co/Zeyue7/VidMuse/">[HF Model 🤗]</a> 
"""

block_css = """
.gradio-container {margin: 0.1% 1% 0 1% !important; max-width: 98% !important;};
#buttons button {
    min-width: min(120px,100%);
}

.gradient-text {
    font-size: 28px;
    width: auto;
    font-weight: bold;
    background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
}

.plain-text {
    font-size: 22px;
    width: auto;
    font-weight: bold;
}
"""


def predict_full(model_path, video, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    global USE_DIFFUSION
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    model_path = model_path.strip()
    if model_path:
        if not Path(model_path).exists():
            raise gr.Error(f"Model path {model_path} doesn't exist.")
        if not Path(model_path).is_dir():
            raise gr.Error(f"Model path {model_path} must be a folder containing "
                           "state_dict.bin and compression_state_dict_.bin.")
        model = model_path
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    USE_DIFFUSION = False
    assert model is not None, "The model is not defined. Please enter the model's directory (e.g., './model')"

    load_model(model)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    duration = 10
    videos, wavs = _do_predictions(
        [video], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef,
        gradio_progress=progress)

    return videos[0], wavs[0], None, None


def toggle_diffusion(choice):
    return [gr.update(visible=False)] * 2

def update_video(video_file):
    return video_file
    
def ui_full(launch_kwargs):
    with gr.Blocks(
        title="VidMuse",
        theme=gr.themes.Default(),
        css=block_css
    ) as interface:

        gr.HTML(title_html)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        video = gr.Video(sources=["upload"], label="File",
                                        interactive=True, elem_id="video-input", height=220, width=320)
                
                with gr.Row():
                    model_path = gr.Text(label="Model Folder", value='./model')


                with gr.Accordion("Parameters", open=False) as parameter_row:
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)

                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)

                with gr.Row():
                    gr.Examples(examples=[
                        ["./dataset/example/infer/sample.mp4"],
                        ["./dataset/example/infer/sample_long_video.mp4"],
                    ], inputs=[video])

            with gr.Column(scale=8):
                output = gr.Video(label="VidMuse", height=500, width=960)
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath', scale=0.9)

        submit.click(toggle_diffusion, queue=False,
                     show_progress=False).then(predict_full, inputs=[model_path, video, topk, topp,
                                                                     temperature, cfg_coef],
                                               outputs=[output, audio_output])

        video.change(fn=update_video, inputs=[video], outputs=[video])

        interface.load(js=js,)
        interface.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    ui_full(launch_kwargs)
