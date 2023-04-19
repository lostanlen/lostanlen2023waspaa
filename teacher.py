from nnAudio.features.cqt import CQT
import numpy as np
import pytorch_lightning as pl
import torch

HYPERPARAMS = {
    "speech": {
        "n_filters": 40,
        "win_length": 1024,
        "stride": 256,
        "sr": 44100,
        "fmin": 50, 
        "fmax": 22050,
    },
    "music": {},
    "urban": {},
    "synth": {
        "n_filters": 8,
        "win_length": 2**10,
        "stride": 2**8,
        "sr": 44100,
        "fmin": 50,
        "fmax": 12800,
        "N": 2**13,
        "n_samples": 1000,
    },
}


class SpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, sav_dir, domain):
        super().__init__()
        self.sav_dir = sav_dir
        self.domain = domain
        self.num_workers = 0

    def setup():
        pass


class CQTSineData(torch.utils.data.Dataset):
    def __init__(self, spec, n_samples):
        super().__init__()
        self.spec = spec
        self.cqt_params = {
            "bins_per_octave": spec["bins_per_octave"],
            "fmin": spec["fmin"],
            "hop_length": spec["stride"],
            "n_bins": spec["n_filters"],
            "sr": spec["sr"],
        }
        self.N = spec["N"]
        self.freqs = np.logspace(
            np.log10(spec["fmin"]/2),
            np.log10(spec["fmax"]*2),
            n_samples,
            endpoint=False
        )
        self.closure = CQT(**self.cqt_params)

    def __getitem__(self, freq_id):
        freq = self.freqs[freq_id]
        t = torch.arange(0, self.N/self.spec["sr"], 1/self.spec["sr"])
        x = torch.sin(2*np.pi*freq*t)
        x = x.reshape(1, -1)
        return {'freq': freq, 'x': x, 'Y': self.closure(x)}
    
    def __len__(self):
        return len(self.freqs)


class CQTSineDataModule(pl.LightningDataModule):
    def __init__(self, domain, batch_size):
        super().__init__()
        self.domain = domain
        self.num_workers = 0
        self.batch_size = batch_size

    def setup(self, stage=None):
        spec = HYPERPARAMS[self.domain]
        n_samples = spec["n_samples"]
        dataset = CQTSineData(spec, n_samples)
        n_train = int(0.8*n_samples)
        n_val = int(0.1*n_samples)
        n_test = n_samples - n_train - n_val
        split = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
        self.train_dataset, self.val_dataset, self.test_dataset = split

    def collate_batch(self, batch):
        freq = torch.cat([item['freq'] for item in batch])
        x = torch.cat([item['x'] for item in batch])
        Y = torch.cat([item['Y'] for item in batch])
        return {'freq': freq, 'x': x, 'Y': Y}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
