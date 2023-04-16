import pytorch_lightning as pl

class SpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, sav_dir, domain):
        super().__init__()
        self.sav_dir = sav_dir
        self.domain = domain
        self.num_workers = 0

    def setup():
        pass