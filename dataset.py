"""
Torch dataset object for aligned audio pairs used in voice conversion training.
"""

import os
import glob
import torch
from scipy.io.wavfile import read


def get_dataset(dir):
    original_files = sorted(glob.glob(os.path.join(dir, "*_original.wav")))
    converted_files = [
        f.replace("_original.wav", "_converted.wav") for f in original_files
    ]
    return original_files, converted_files


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


class LLVCDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dir,
        sr,
        wav_len,
        dset
    ):
        assert dset in [
            "train",
            "val",
            "dev"
        ], "`dset` must be one of ['train', 'val', 'dev']"
        self.dset = dset
        file_dir = os.path.join(dir, dset)
        self.wav_len = wav_len
        self.sr = sr
        self.original_files, self.converted_files = get_dataset(
            file_dir
        )
        # In-memory cache to avoid redundant disk reads after the first epoch
        self._cache = {}

    def __len__(self):
        return len(self.original_files)

    def _load_and_normalize(self, path):
        data, sr = load_wav(path)
        assert sr == self.sr, f"Expected {self.sr}Hz, got {sr}Hz for file {path}"
        tensor = torch.from_numpy(data).unsqueeze(0).float() / 32768.0
        return tensor

    def _pad_or_trim(self, tensor):
        if tensor.shape[-1] < self.wav_len:
            tensor = torch.nn.functional.pad(
                tensor, (0, self.wav_len - tensor.shape[-1]))
        else:
            tensor = tensor[:, :self.wav_len]
        return tensor

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        original = self._load_and_normalize(self.original_files[idx])
        converted = self._load_and_normalize(self.converted_files[idx])

        original = self._pad_or_trim(original)
        gt = self._pad_or_trim(converted)

        self._cache[idx] = (original, gt)
        return original, gt
