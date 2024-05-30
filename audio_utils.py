import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def split_audio(audio_file, segment_duration=1, save_output_dir: str = None, target_sr=22050):

    y, sr = librosa.load(audio_file, sr=target_sr)

    segment_samples = int(sr * segment_duration)

    num_segments = int(len(y) // segment_samples)
    segment_list = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = y[start:end]
        segment_list.append((segment, sr))
        if save_output_dir is not None:
            os.makedirs(save_output_dir, exist_ok=True)
            segment_output_file = os.path.join(save_output_dir, f"seg_{i}.wav")
            sf.write(segment_output_file, segment, sr)
    return segment_list


def remove_silent_segment(segment_list, threshold_db=-40):
    removed_samples = 0
    new_segment_list = []
    for segment in segment_list:
        y, sr = segment
        rms = librosa.feature.rms(y=y)
        db = 20 * np.log10(rms)
        db = np.mean(db)
        if db >= threshold_db:
            new_segment_list.append(segment)
        else:
            removed_samples += 1
    return new_segment_list, removed_samples


def normalize_mfcc(mfcc):
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    normalized_mfcc = (mfcc - mean) / std
    return normalized_mfcc


def extract_mfcc_segment(audio_segment, n_mfcc=20):
    y, sr = audio_segment
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.array(mfcc)


def plot_mfcc_segments(title, mfccs):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()
