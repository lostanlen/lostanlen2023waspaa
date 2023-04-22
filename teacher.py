from nnAudio.features.cqt import CQT
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import torch
import os

HYPERPARAMS = {
    "speech": { #mel
        "n_filters": 42,
        "nfft": 32000,
        "win_length": 1024,
        "stride": 256,
        "sr": 16000, 
        "fmin": 65, 
        "fmax": 7696.2,
    },
    "music": { #vqt
        "n_filters": 96,
        "nfft": 88200,
        "win_length": None, #TODO: add sr
        "fmin": 100,
        "fmax": 22050,
    }, 
    "urban": {  #third octave
        "n_filters": 32,
        "nfft": 88200,
        "win_length": None, #different for each filter
        "sr": 44100,
        "fmin": 25,
        "fmax": 20000,
    },
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
  

class SpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, sav_dir, domain, batch_size):
        super().__init__()
        self.audio_dir = os.path.join(sav_dir, domain + ".h5")
        self.batch_size = batch_size
        if domain == "speech":
            self.feature = "mel"
        elif domain == "music":
            self.feature = "cqt"
        elif domain == "urban":
            self.feature = "third_oct_response"
        self.seg_length = HYPERPARAMS[self.feature]["nfft"]
        self.stride = HYPERPARAMS[self.feature]["stride"]
        self.num_workers = 0
    
    def setup(self, stage=None):
        self.train_dataset = SpectrogramData(self.audio_dir, self.feature, self.seg_length, self.stride)
        self.val_dataset = SpectrogramData(self.audio_dir, self.feature, self.seg_length, self.stride)
        

    def collate_batch(self, batch):
        feat = torch.stack([s['feature'] for s in batch], axis=0)
        x = torch.stack([s['x']for s in batch], axis = 0).permute(0,2,1) #batch, time, channel
        #print("batched feature shape", feat.shape, x.shape)
        return {'feature': feat, 'x': x}


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)

class SpectrogramData(Dataset):
    def __init__(self,
                 audio_dir,
                 feature,
                 seg_length,
                 stride,
        ):
        super().__init__()
        self.feature = feature
        self.audio_dir = audio_dir #path to hdf5 file
        self.ids = self.get_ids()
        self.seg_length = seg_length
        self.stride = stride
        #load filter coefficients:
        coef_path = os.path.join(audio_dir, self.feature + "_freqz.npy") #store coefficeints in "mel_freqz.npy"
        self.coefficients = np.load(coef_path)

    def __getitem__(self, idx): 
        id = self.ids[idx]
        x, feat = self.feat_from_id(id)
        return {'feature': feat, 'x': x}

    def __len__(self):
        return len(self.ids)
    
    def get_ids(self):
        with h5py.File(self.audio_dir, "r") as f:
            length = len(f['x'].keys())
        return list(str(i) for i in np.arange(1, length + 1))

    def feat_from_id(self, id):
        with h5py.File(self.audio_dir, "r") as f:
            x = np.array(f['x'][str(id)])
        #sample a random segment 
        start = np.random(x.shape[0] - self.seg_length)
        x = torch.tensor(x[start: start+self.seg_length], dtype=torch.float32).cuda()
        feat = filtering(x, self.coefficients, self.stride)
        return x, feat
    

def filtering(x, freqz, stride):
    """
    x: (time, )
    freqz: (nfft, n_filters)
    """
    #take filter coefficients and convolve with x
    #take fft of x
    N = x.shape[0]
    x_fft = torch.fft.fft(x) #(nfft)
    assert freqz.shape[0] == x_fft.shape[0]
    #complex pointwise multiplication with freqz
    y = torch.fft.ifft(x_fft[:,None] * freqz, dim=0) #(nfft, n_filters)
    #subsample in fourier domain equivalent to strided convolution
    y = y[torch.arange(0, N, stride), :] #(frequency, n_filters)
    return y.T #(n_filters, frequency)
