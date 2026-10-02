"""Model zoo: TRF baseline, CNN (AADNet), Graph (ST-GCN), VLAAI, StimulusGAT."""
from .aadnet import AADNet
from .stgcn import STGCN
from .trf_baseline import TRFDecoder
from .vlaai_pytorch import VLAAIPyTorch
from .vlaai_decision import AADDecisionWrapper, AADDecisionEEGOnly
from .trf_decision import TRFDecisionWrapper
from .sgat import StimulusGAT
from .sgat_decision import StimulusGATDecisionWrapper

__all__ = [
    "AADNet", "STGCN", "TRFDecoder", "TRFDecisionWrapper",
    "VLAAIPyTorch", "AADDecisionWrapper", "AADDecisionEEGOnly",
    "StimulusGAT", "StimulusGATDecisionWrapper",
]
