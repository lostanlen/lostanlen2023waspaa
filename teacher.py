import pytorch_lightning as pl

HYPERPARAMS = {
    "speech": {
        "n_filters": 40,
        "win_length": 1024,
        "stride": 256,
        "sr": 44100, ##TODO: add sr
        "fmin": 50, 
        "fmax": 22050,
    },
    "music": {},
    "urban": {},
}


class SpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, sav_dir, domain):
        super().__init__()
        self.sav_dir = sav_dir
        self.domain = domain
        self.num_workers = 0

    def setup():
        pass
