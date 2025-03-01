import os
import json
import librosa

wav_dir = './dataset/example/train/audio'
mp4_dir = './dataset/example/train/video'

output_file = './egs/V2M/example/data.jsonl'
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

with open(output_file, 'w') as f_out:
    for wav_file in os.listdir(wav_dir):
        if wav_file.endswith('.wav'):
            ytb_id = os.path.splitext(wav_file)[0]
            
            wav_path = os.path.join(wav_dir, wav_file)
            mp4_path = os.path.join(mp4_dir, ytb_id + '.mp4')
            
            wav_path_abs = os.path.abspath(wav_path)
            mp4_path_abs = os.path.abspath(mp4_path)
            
            audio, sr = librosa.load(wav_path_abs, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)

            info_dict = {
                "path": wav_path_abs,
                "video_path": mp4_path_abs,
                "duration": duration,
                "sample_rate": sr,
                "amplitude": None,
                "weight": None,
                "info_path": None
            }
            
            f_out.write(json.dumps(info_dict) + '\n')
