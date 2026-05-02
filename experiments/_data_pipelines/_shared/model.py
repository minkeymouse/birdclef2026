"""SED model architecture."""
import timm
import torch
import torch.nn as nn
import torchaudio

from .constants import (SR, N_FFT, HOP, N_MELS, F_MIN, F_MAX,
                          SPEC_FREQ_MASK, SPEC_TIME_MASK, BACKBONE)


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=F_MIN, f_max=F_MAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x):
        return self.adb(self.mel(x)).unsqueeze(1)


class SpecAug(nn.Module):
    def __init__(self, f=SPEC_FREQ_MASK, t=SPEC_TIME_MASK):
        super().__init__()
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=f)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=t)

    def forward(self, x):
        return self.tm(self.fm(x))


class SEDHead(nn.Module):
    def __init__(self, feat_dim, n_cls):
        super().__init__()
        self.att = nn.Conv1d(feat_dim, n_cls, 1)
        self.cla = nn.Conv1d(feat_dim, n_cls, 1)

    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values


class SEDModel(nn.Module):
    def __init__(self, n_cls, backbone=BACKBONE, pretrained=True):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.specaug = SpecAug()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=1,
                                            num_classes=0, global_pool='')
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(f.shape[1], n_cls)

    def forward(self, x, train=True):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        if train:
            m = self.specaug(m)
        f = self.backbone(m)
        f = f.mean(dim=2) if f.dim() == 4 else f
        clip, fmax = self.head(f)
        return clip, fmax
