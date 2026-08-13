"""Real-world dataset module stub for insertion tasks."""

import lightning as L
import omegaconf


class RealWorldDataModule(L.LightningDataModule):
    """Placeholder for the real-world insertion dataset (not needed for RPDiff reproduction)."""

    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg=None):
        super().__init__()
        raise NotImplementedError(
            "RealWorldDataModule is not implemented in this release. "
            "It is only needed for the NIST insertion task."
        )
