import argparse
from pathlib import Path
import gradio as gr
from demos.VidMuse_app import load_model, _do_predictions

def infer(video_dir, output_dir, topk, topp, temperature, cfg_coef):
    # Get all mp4 files in the directory
    video_files = list(Path(video_dir).glob('*.mp4'))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for video_path in video_files:
        # Perform predictions and save generated files to the specified output directory
        output_video_path = Path(output_dir) / f"{video_path.stem}_output.mp4"
        output_audio_path = Path(output_dir) / f"{video_path.stem}_output.wav"
        
        # Skip if the output audio file already exists
        if output_audio_path.exists():
            print(f"Audio file already exists, skipping: {output_audio_path}")
            continue        
                
        videos, wavs = _do_predictions(
            [str(video_path)], duration=30, progress=False,
            top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef
        )

        # Assume the paths returned by _do_predictions are temporary and need to be moved to output_dir
        Path(videos[0]).rename(output_video_path)
        Path(wavs[0]).rename(output_audio_path)

        print(f"Generated video file: {output_video_path}")
        print(f"Generated audio file: {output_audio_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VidMuse inference script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the input video directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--topk', type=int, default=250, help='Top-k parameter')
    parser.add_argument('--topp', type=float, default=0.0, help='Top-p parameter')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter')
    parser.add_argument('--cfg_coef', type=float, default=3.0, help='Classifier-free guidance coefficient')

    args = parser.parse_args()
    
    load_model(args.model_path)
    infer(args.video_dir, args.output_dir, args.topk, args.topp, args.temperature, args.cfg_coef)