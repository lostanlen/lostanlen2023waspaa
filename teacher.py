from nnAudio.features.cqt import CQT
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import torch
import os
import soundfile as sf
import torch.nn.functional as F
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

# load everything about the teachers as dictionaries, containing:
#   "freqz" frequency responses as 2D array (time,channels)
#   "centerfreq" 
#   "bandwidths"
#   "framebounds" computed without subsampling
with open('Freqz/GAM.pkl', 'rb') as fp:
    GAM = pickle.load(fp)
with open('Freqz/VQT.pkl', 'rb') as fp:
    VQT = pickle.load(fp)
with open('Freqz/THIRD.pkl', 'rb') as fp:
    THIRD = pickle.load(fp)


HYPERPARAMS = {
    "speech_gam": { #gam
        "n_filters": 67,
        "seg_length": 32000,
        "win_length": 1024,
        "stride": 256,
        "a_opt": 8,
        "sr": 16000, 
        "fmin": 12.5, 
        "fmax": 8000,
        "octaves": GAM["octave_tags"],
    },
    "music": { #vqt
        "n_filters": 96,
        "seg_length": 2**16,
        "win_length": 1024, 
        "stride": 256,
        "a_opt": 16,
        "sr": 44100,
        "fmin": 100,
        "fmax": 22050,
        "octaves": VQT["octave_tags"],
    }, 
    "urban": {  #third
        "n_filters": 32,
        "seg_length": 88200,
        "win_length": 1024,
        "stride": 256,
        "a_opt": 8,
        "sr": 44100,
        "fmin": 25,
        "fmax": 20000,
        "octaves": THIRD["octave_tags"],
    },
    "synth": {
        "bins_per_octave": 1,
        "n_filters": 8,
        "win_length": 2**10,
        "stride": 2**8,
        "sr": 44100,
        "fmin": 50,
        "fmax": 12800,
        "seg_length": 2**14,
        "n_samples": 1000,
        "octaves": reversed(range(8)),
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
        self.seg_length = spec["seg_length"]
        self.freqs = np.logspace(
            np.log10(spec["fmin"]/2),
            np.log10(spec["fmax"]*2),
            n_samples,
            endpoint=False
        )
        self.closure = CQT(**self.cqt_params)

    def __getitem__(self, freq_id):
        freq = self.freqs[freq_id]
        t = torch.arange(
            0, self.seg_length/self.spec["sr"], 1/self.spec["sr"])
        x = torch.sin(2*np.pi*freq*t)
        x = x.reshape(1, -1)
        return {'freq': freq, 'x': x, 'feature': self.closure(x)}
    
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
        self.audio_dir = sav_dir
        self.batch_size = batch_size
        self.ids, self.file_names = self.get_ids()
        if domain == "speech":
            self.feature = "mel"
            self.coef_dir = 'Freqz/GAM.pkl'
        elif domain == "music":
            self.feature = "vqt"
            self.coef_dir = 'Freqz/VQT.pkl'
        elif domain == "urban":
            self.feature = "third_oct_response"
            self.coef_dir = 'Freqz/THIRD.pkl'
        self.seg_length = HYPERPARAMS[domain]["seg_length"]
        self.stride = HYPERPARAMS[domain]["stride"]
        self.num_workers = 0

    def setup(self, stage=None):
        N = len(self.ids)
        train_ids = self.ids[:-N//5]
        test_ids = self.ids[-N // 5: -N // 10]
        val_ids = self.ids[-N//10::]
        self.train_dataset = SpectrogramData(train_ids, self.audio_dir, self.coef_dir, self.file_names, self.feature, self.seg_length, self.stride)
        self.val_dataset = SpectrogramData(test_ids, self.audio_dir, self.coef_dir, self.file_names, self.feature, self.seg_length, self.stride)
        self.test_dataset = SpectrogramData(val_ids, self.audio_dir, self.coef_dir, self.file_names, self.feature, self.seg_length, self.stride)
    
    def get_ids(self):
        files = []
        for root, dirs, fs in os.walk(self.audio_dir):
            for f in fs:
                if f[-3:] == "wav":
                    files.append(os.path.join(root, f))
        length = len(files)
        return list(i for i in np.arange(1, length + 1)), files

    def collate_batch(self, batch):
        feat = torch.stack([s['feature'] for s in batch], axis=0)
        x = torch.stack([s['x']for s in batch]) #batch, time, channel
        #print("batched feature shape", feat.shape, x.shape, feat.dtype)
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
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)

class SpectrogramData(Dataset):
    def __init__(self,
                 ids,
                 audio_dir,
                 coef_dir,
                 file_names, 
                 feature,
                 seg_length,
                 stride,
        ):
        super().__init__()
        self.feature = feature
        self.audio_dir = audio_dir #path to hdf5 file
        self.file_names = file_names
        self.seg_length = seg_length
        self.stride = stride
        self.ids = ids
        #load filter coefficients:
        with open(coef_dir, 'rb') as fp:
            self.freqz = pickle.load(fp)['freqz']

    def __getitem__(self, idx): 
        id = self.ids[idx]
        x, feat = self.feat_from_id(id)
        return {'feature': feat, 'x': x}

    def __len__(self):
        return len(self.ids)

    def feat_from_id(self, id):
        x, sr = sf.read(self.file_names[int(id)])
        #sample a random segment 
        start = np.random.randint(x.shape[0] - self.seg_length - 1)
        x = torch.tensor(x[start: start+self.seg_length], dtype=torch.float32)#.to(device)
        feat = filtering_time(x, self.freqz, self.stride)
        return x, feat
    

# Teacher filtering routines:

def filtering_frequency(x, freqz, stride):
    """
    x: (sig_length, )
    freqz: (sig_length, n_filters)

    output: filterbank magnitude responses for x (n_filters, sig_length_subsampled)
    """
    
    freqz = torch.from_numpy(freqz)
    N = x.shape[0]

    # x to Fourier domain
    x_fft = torch.fft.fft(x) #(nfft)
    assert freqz.shape[0] == x_fft.shape[0]

    # complex pointwise multiplication with freqz and ifft with renormalization
    y = torch.fft.ifft(x_fft * freqz, dim=0)*freqz.shape[0] #(nfft, n_filters)

    # strided convolution = subsampling the full convolution
    y = y[torch.arange(0, N - np.mod(N,stride), stride), :].T #(n_filters, nfft_subsampled)

    # magnitude
    mag = torch.tensor(torch.imag(y) ** 2 + torch.real(y) ** 2, dtype=torch.float) #(frequency, n_filters)
    return mag


def get_imp(freqz, centering = True, to_torch = True):
    """
    Get the impulse reponses as tensors

    freqz: (sig_length, n_filters)

    output: centered impulse reponses (n_filters, sig_length)
    """

    # to time-domain --> impulse reponse
    imp = np.fft.ifft(freqz,axis=0)*np.sqrt(freqz.shape[0])
    imp_r = np.real(imp)
    imp_i = np.imag(imp)

    # center the impulse responses within the vector instead of at the origin/end
    if centering:
        imp_r = np.roll(imp_r,len(imp_r)//2,axis = 0).T
        imp_i = np.roll(imp_i,len(imp_i)//2,axis = 0).T
    if to_torch:
        imp_r = torch.from_numpy(imp_r)
        imp_i = torch.from_numpy(imp_i)
    return imp_r, imp_i


def filtering_time(x, freqz, stride):
    """
    Do the filtering with conv1D.

    x: (sig_length, )
    freqz: (sig_length, n_filters)

    output: filterbank magnitude responses for x (n_filters, sig_length_subsampled)
    """

    # some shaping to fit for conv1D
    x = x.T
    x = x.unsqueeze(0)
    imp_r, imp_i = get_imp(freqz) # remove lowpass filter 
    imp_rt = imp_r.unsqueeze(1).float()
    imp_it = imp_i.unsqueeze(1).float()

    # with padding=freqz.shape[0]//2 we start with the center of the filter at the first signal entry 
    # in this way we are close to the circulant setting, but done with conv1D
    out_r = F.conv1d(x, imp_rt, bias=None, stride=stride, padding=freqz.shape[0]//2)
    out_i = F.conv1d(x, imp_it, bias=None, stride=stride, padding=freqz.shape[0]//2)
    # magnitude
    mag = out_r ** 2 + out_i ** 2
    return mag
