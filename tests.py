import os
import unittest
import numpy as np
from audio_utils import split_audio, extract_mfcc_segment
from data_utils import create_dataset


class TestAudioUtils(unittest.TestCase):

    def test_split_audio(self):
        for file_name in os.listdir("data"):
            segments = split_audio("data/" + file_name, segment_duration=1)
            segments_shape = list(map(lambda x: x[0].shape == segments[0][0].shape, segments))
            self.assertEqual(all(segments_shape), True)

    def test_extract_mfcc(self):
        # Create a dummy audio segment
        sr = 22050
        t = np.linspace(0, 1, sr, endpoint=False)  # 1 second
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        segment = (y, sr)

        mfcc = extract_mfcc_segment(segment, n_mfcc=20)
        self.assertEqual(mfcc.shape[0], 20)  # 20 MFCC
        self.assertGreater(mfcc.shape[1], 0)  # Ensure there are time frames
