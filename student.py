from collections import Counter
from murenn import DTCWTForward
import numpy as np
import torch
import torchaudio
import torch.nn as nn
from speechbrain.nnet.CNN import GaborConv1d
import pytorch_lightning as pl
import torch.nn.utils.parametrize as P
import torch.nn.functional as F

class TDFilterbank(pl.LightningModule):
    def __init__(self, spec):
        super().__init__()

        self.psi_real = torch.nn.Conv1d(
            in_channels=1,
            out_channels=spec["n_filters"],
            kernel_size=spec["win_length"],
            stride=spec["stride"],
            padding=spec["win_length"]//2,
            bias=False,
        )

        self.psi_imag = torch.nn.Conv1d(
            in_channels=1,
            out_channels=spec["n_filters"],
            kernel_size=spec["win_length"],
            stride=spec["stride"],
            padding=spec["win_length"]//2,
            bias=False,
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        Wx_real = self.psi_real(x)
        Wx_imag = self.psi_imag(x)
        Ux = Wx_real * Wx_real + Wx_imag * Wx_imag
        return Ux

    def step(self, batch, fold):
        feat = batch['feature'].squeeze()#.to(self.device)
        x = batch['x']#.to(self.device).double()
        outputs = self(x)
        loss = F.mse_loss(outputs[:,1:,:], feat[:,1:,:]) 
        return {'loss': loss}
    
    def training_step(self, batch):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=False)
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, prog_bar=False)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
       

class MuReNN(pl.LightningModule):
    def __init__(self, spec, Q_multiplier=16):
        super().__init__()
        
        mel_scale = "htk"
        m_min = torchaudio.functional.functional._hz_to_mel(
            spec["fmin"], mel_scale=mel_scale)
        m_max = torchaudio.functional.functional._hz_to_mel(
            spec["fmax"], mel_scale=mel_scale)
        m_pts = torch.linspace(m_min, m_max, spec["n_filters"] + 2)
        f_pts = torchaudio.functional.functional._mel_to_hz(
            m_pts, mel_scale=mel_scale)
        center_freqs = f_pts[1:-1]

        nyquist = spec["sr"] / 2
        octaves = np.round(np.log2(
            nyquist / center_freqs.numpy())).astype("int")
        Q_ctr = Counter(octaves)
        self.J_psi = max(Q_ctr)
        self.stride = spec["stride"]
        
        self.tfm = DTCWTForward(J=1+self.J_psi,
            alternate_gh=True, include_scale=False)

        psis = []
        for j in range(1+self.J_psi):
            kernel_size = Q_multiplier*Q_ctr[j]
            if j == 0:
                stride_j = spec["stride"]
            else:
                stride_j = spec["stride"] // (2**(j-1))
            psi = torch.nn.Conv1d(
                in_channels=1,
                out_channels=Q_ctr[j],
                kernel_size=kernel_size,
                stride=stride_j,
                bias=False,
                padding=kernel_size//2)
            psis.append(psi)
            
        self.psis = torch.nn.ParameterList(psis)
                  
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        _, x_levels = self.tfm.forward(x)
        Ux = []
        
        for j_psi in range(1+self.J_psi):
            x_level = x_levels[j_psi].type(torch.complex64) / (2**j_psi)
            Wx_real = self.psis[j_psi](x_level.real)
            Wx_imag = self.psis[j_psi](x_level.imag)
            Ux_j = Wx_real * Wx_real + Wx_imag * Wx_imag
            Ux_j = torch.real(Ux_j) 
            Ux.append(Ux_j)

        Ux = torch.cat(Ux, axis=2)

        # Flip j axis so that frequencies range from low to high
        Ux = torch.flip(Ux, dims=(-2,))
        return Ux
    
    def step(self, batch, fold):
        feat = batch['feature'].squeeze()#.to(self.device)
        x = batch['x']#.to(self.device).double()
        outputs = self(x)
        loss = F.mse_loss(outputs[:,1:,:], feat[:,1:,:]) 
        return {'loss': loss}
    
    def training_step(self, batch):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=False)
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, prog_bar=False)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
       
class Exp(nn.Module):
    def forward(self, X):
        return torch.exp(X)

class Leaf(pl.LightningModule):
    def __init__(self, spec, learn_amplitudes=False):
        super().__init__()
        self.learn_amplitudes = learn_amplitudes
        self.gaborfilter = GaborConv1d(
            out_channels=2*spec['n_filters'], 
            kernel_size=spec['win_length'], #filter length: should be the provided freqz length divided by stride size?
            stride=spec['stride'],
            input_shape=None,
            in_channels=spec['n_filters'],
            padding='same',
            padding_mode='constant',
            sample_rate=spec['sr'],
            min_freq=spec['fmin'],
            max_freq=spec['fmax'],
            n_fft=spec['win_length'], #construct mel filters and determine initialization of gabor filters
            normalize_energy=False,
            bias=False,
            sort_filters=True, #ascending order 
            use_legacy_complex=True,
            skip_transpose=False, #false means enable batch processing
        )
        self.learnable_scaling = nn.Conv1d(
            in_channels=spec['n_filters'],
            out_channels=spec['n_filters'],
            kernel_size=1,
            groups=spec['n_filters'],
            bias=False
        )
        # Ensure positiveness of learned parameters
        #P.register_parametrization(self.learnable_scaling, "weight", Exp()) 

    def forward(self, x): 
        Ux = self.gaborfilter(x) #(batch, time, filters)
        #print("gabor what shape", Ux.shape, Ux.dtype, torch.sum(Ux<0))
        if self.learn_amplitudes: 
            #apply learnable scaling to corresponding real and imaginary channels
            Ux[:,:,:Ux.shape[-1]//2] = self.learnable_scaling(Ux[:,:,:Ux.shape[-1]//2])  
            Ux[:,:,Ux.shape[-1]//2::] = self.learnable_scaling(Ux[:,:,Ux.shape[-1]//2::]) 
        #Ux shape (batch, time, 2*n_filters)
        n_filters = Ux.shape[-1]
        mag = Ux[:,:,:n_filters//2] ** 2 + Ux[:,:,n_filters//2::] ** 2
        return mag.permute(0,2,1)

    def step(self, batch, fold):
        feat = batch['feature']#.to(self.device)
        x = batch['x']#.to(self.device).double()
        outputs = self(x)
        loss = F.mse_loss(outputs[:,1:,:], feat[:,1:,:]) #remove the low pass filter from loss
        return {'loss': loss}
    
    def training_step(self, batch):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=False)
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, prog_bar=False)
        

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
       