from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile as sf
import librosa
import os

def split_wav_by_silence(input_path, output_path, min_silence_len=2000, silence_thresh=-40, starting_index=1):
    audio = AudioSegment.from_wav(input_path)
    
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=500)

    for i, chunk in enumerate(chunks):
        chunk.export(f"{output_path}/{starting_index + i}.wav", format="wav")

def resample_wav(input_file, output_file, target_sr):
    data, sr = librosa.load(input_file, sr=None)
    data_resampled = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    sf.write(output_file, data_resampled, target_sr)

def process_wav_folder(input_folder, output_folder, target_sr, starting_index=1):
    for i, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith('.wav'):
            filename = f"{i + starting_index}.wav"
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{i + start_index}.wav")
            resample_wav(input_file, output_file, target_sr)

def count_files_in_folder(folder_path):
    file_count = 0
    for _, _, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

input_path = ".\\test12.wav"
output_path = ".\\output3"
start_index = count_files_in_folder(".\\wavs") + 1

split_wav_by_silence(input_path, output_path, starting_index=start_index)

target_sr = 22050
input_path = '.\\output3'
output_path = '.\\wavs2'

process_wav_folder(input_path, output_path, target_sr, start_index)