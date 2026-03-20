"""
Medical Image Dataset Loaders
==============================
Unified PyTorch Dataset classes for the three benchmarks used in the FCGD paper:
  • MMWHS  – Multi-Modality Whole Heart Segmentation (CT ↔ MRI)
  • CHAOS  – Combined CT-MR Abdominal Organ Segmentation
  • MS-CMRSeg – Multi-sequence CMR (bSSFP ↔ LGE)

All datasets:
  • Resize to 256×256
  • Normalise to [0, 1] (z-score per volume)
  • Return (image, label) pairs for source; (image,) tuples for unlabelled target
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Optional, Callable, List, Tuple
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalise_volume(vol: np.ndarray) -> np.ndarray:
    """Z-score normalise a 3-D volume to zero mean, unit std."""
    mu, sigma = vol.mean(), vol.std()
    return (vol - mu) / (sigma + 1e-8)


def resize_slice(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize a 2-D slice (H×W) to size×size using bilinear interpolation."""
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def resize_mask(msk: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize an integer mask using nearest-neighbour interpolation."""
    t = torch.from_numpy(msk).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode='nearest')
    return t.squeeze().long().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Base 2-D Slice Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SliceDataset(Dataset):
    """
    Generic 2-D axial-slice dataset built from a list of (slice, label) pairs.

    Args:
        slices      : list of (image_array, label_array) or (image_array,)
        transform   : optional augmentation callable applied to image tensor
        labelled    : whether labels are available
        img_size    : target spatial resolution
    """

    def __init__(
        self,
        slices: List,
        transform: Optional[Callable] = None,
        labelled: bool = True,
        img_size: int = 256,
    ):
        self.slices   = slices
        self.tf       = transform
        self.labelled = labelled
        self.size     = img_size

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int):
        entry = self.slices[idx]
        img = resize_slice(entry[0], self.size)
        img_t = torch.from_numpy(img).float().unsqueeze(0)   # (1, H, W)
        if self.tf is not None:
            img_t = self.tf(img_t)

        if self.labelled and len(entry) > 1:
            lbl = resize_mask(entry[1], self.size)
            lbl_t = torch.from_numpy(lbl).long()              # (H, W)
            return img_t, lbl_t
        return (img_t,)


# ─────────────────────────────────────────────────────────────────────────────
# MMWHS Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MMWHSDataset(SliceDataset):
    """
    Multi-Modality Whole Heart Segmentation (MMWHS) dataset.

    Expects the standard MMWHS directory structure:
        root/
          ct_train/   image_*.nii.gz  label_*.nii.gz
          mr_train/   image_*.nii.gz  label_*.nii.gz
          ct_test/    …
          mr_test/    …

    Label map: 0=BG, 1=MYO, 2=LAC, 3=LVC, 4=AA

    Args:
        root    : dataset root directory
        modality: 'ct' or 'mr'
        split   : 'train' or 'test'
        labelled: use labelled (train) or unlabelled (test) mode
        img_size: target image size
    """

    LABEL_MAP = {0: 0, 500: 1, 420: 2, 205: 3, 820: 4}

    def __init__(
        self,
        root:     str,
        modality: str = 'ct',
        split:    str = 'train',
        labelled: bool = True,
        img_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required: pip install nibabel")

        folder = os.path.join(root, f'{modality}_{split}')
        img_paths = sorted(glob.glob(os.path.join(folder, 'image_*.nii.gz')))
        lbl_paths = sorted(glob.glob(os.path.join(folder, 'label_*.nii.gz')))

        slices = []
        for ip, lp in zip(img_paths, lbl_paths):
            vol  = normalise_volume(nib.load(ip).get_fdata())
            mask = nib.load(lp).get_fdata().astype(np.int32)
            # Remap label values
            remapped = np.zeros_like(mask)
            for orig, new in self.LABEL_MAP.items():
                remapped[mask == orig] = new
            for s in range(vol.shape[-1]):      # iterate axial slices
                slices.append((vol[..., s], remapped[..., s]))

        super().__init__(slices, transform=transform,
                         labelled=labelled, img_size=img_size)


# ─────────────────────────────────────────────────────────────────────────────
# CHAOS Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CHAOSDataset(SliceDataset):
    """
    CHAOS (Combined CT-MR Healthy Abdominal Organ Segmentation) dataset.

    Expects CHAOS DICOM/NIfTI folder structure:
        root/
          CT/   patient_*/  Ground/  DICOM_anon/
          MR/   patient_*/  T1DUAL/  T2SPIR/

    Label map: 55=liver, 110=right_kidney, 175=left_kidney, 240=spleen
               → remapped to 1,2,3,4
    """

    LABEL_MAP = {55: 1, 110: 2, 175: 3, 240: 4}

    def __init__(
        self,
        root:     str,
        modality: str = 'CT',
        split:    str = 'train',
        labelled: bool = True,
        train_frac: float = 0.75,
        img_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required: pip install nibabel")

        modal_root = os.path.join(root, modality)
        patients   = sorted(os.listdir(modal_root))
        n_train    = int(len(patients) * train_frac)
        patients   = patients[:n_train] if split == 'train' else patients[n_train:]

        slices = []
        for pid in patients:
            vol_paths = glob.glob(os.path.join(modal_root, pid, '**', '*.nii.gz'),
                                   recursive=True)
            seg_paths = glob.glob(os.path.join(modal_root, pid, 'Ground', '*.nii.gz'),
                                   recursive=True)
            if not vol_paths or not seg_paths:
                continue
            vol  = normalise_volume(nib.load(vol_paths[0]).get_fdata())
            mask = nib.load(seg_paths[0]).get_fdata().astype(np.int32)
            remapped = np.zeros_like(mask)
            for orig, new in self.LABEL_MAP.items():
                remapped[mask == orig] = new
            for s in range(vol.shape[-1]):
                slices.append((vol[..., s], remapped[..., s]))

        super().__init__(slices, transform=transform,
                         labelled=labelled, img_size=img_size)


# ─────────────────────────────────────────────────────────────────────────────
# MS-CMRSeg Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MSCMRDataset(SliceDataset):
    """
    MS-CMRSeg multi-sequence CMR dataset (bSSFP ↔ LGE).

    Expects:
        root/
          bSSFP/  patient_*_C0.nii.gz   patient_*_C0_manual.nii.gz
          LGE/    patient_*_LGE.nii.gz  patient_*_LGE_manual.nii.gz

    Label map: 200=RVC, 500=MYO, 600=LVC → remapped to 1,2,3
    """

    LABEL_MAP = {200: 1, 500: 2, 600: 3}

    def __init__(
        self,
        root:     str,
        sequence: str = 'bSSFP',
        labelled: bool = True,
        img_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required: pip install nibabel")

        seq_root  = os.path.join(root, sequence)
        vol_paths = sorted(glob.glob(os.path.join(seq_root, '*C0*.nii.gz'))
                           + glob.glob(os.path.join(seq_root, '*LGE*.nii.gz')))
        lbl_paths = sorted(glob.glob(os.path.join(seq_root, '*manual*.nii.gz')))
        vol_paths = [p for p in vol_paths if 'manual' not in p]

        slices = []
        for vp, lp in zip(vol_paths, lbl_paths):
            vol  = normalise_volume(nib.load(vp).get_fdata())
            mask = nib.load(lp).get_fdata().astype(np.int32)
            remapped = np.zeros_like(mask)
            for orig, new in self.LABEL_MAP.items():
                remapped[mask == orig] = new
            for s in range(vol.shape[-1]):
                slices.append((vol[..., s], remapped[..., s]))

        super().__init__(slices, transform=transform,
                         labelled=labelled, img_size=img_size)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    dataset_name: str,
    root: str,
    source_modality: str,
    target_modality: str,
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build source (labelled) and target (unlabelled) DataLoaders.

    Args:
        dataset_name    : 'mmwhs' | 'chaos' | 'mscmr'
        root            : dataset root path
        source_modality : modality string for source domain
        target_modality : modality string for target domain
        batch_size      : training batch size
        num_workers     : DataLoader worker processes
        img_size        : target image resolution

    Returns:
        (source_loader, target_loader)
    """
    CLS = {'mmwhs': MMWHSDataset, 'chaos': CHAOSDataset, 'mscmr': MSCMRDataset}
    assert dataset_name.lower() in CLS, f"Unknown dataset: {dataset_name}"
    DS = CLS[dataset_name.lower()]

    kwargs = dict(root=root, img_size=img_size)
    if dataset_name.lower() in ('mmwhs', 'chaos'):
        src_ds = DS(modality=source_modality, split='train', labelled=True,  **kwargs)
        tgt_ds = DS(modality=target_modality, split='train', labelled=False, **kwargs)
    else:  # mscmr
        src_ds = DS(sequence=source_modality, labelled=True,  **kwargs)
        tgt_ds = DS(sequence=target_modality, labelled=False, **kwargs)

    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    return src_loader, tgt_loader
