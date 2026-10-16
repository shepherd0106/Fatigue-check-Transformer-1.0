"""Unimodal CNNs retaining only one immutable initialized EF branch."""
import torch
from torch import nn
from fatigue_training.development import stream_seed
from fatigue_training.development_models import DevelopmentModel,scoped_rng
import fatigue_diagnostics as frozen

COUNTS={"C_E":1802,"C_F":1634}


class ModalityModel(nn.Module):
    def __init__(self,variant,seed=17,fold=2):
        super().__init__()
        if variant not in COUNTS or seed!=17 or fold not in (2,3,4,5):raise ValueError("Unsupported fixed modality model")
        reference=DevelopmentModel("C_EF",seed,fold)
        _,self.reference_components=frozen.initial_components(reference)
        self.variant=variant;self.modality="eeg" if variant=="C_E" else "fnirs"
        self.add_module(self.modality,getattr(reference,self.modality))
        self.dropout=reference.dropout
        with scoped_rng(stream_seed(seed,fold,"modality.head_allocation")):
            self.head=nn.Linear(32,2)
        start=0 if variant=="C_E" else 32
        with torch.no_grad():
            self.head.weight.copy_(reference.head.weight[:,start:start+32]);self.head.bias.copy_(reference.head.bias)
        if not torch.equal(self.head.weight,reference.head.weight[:,start:start+32]) or not torch.equal(self.head.bias,reference.head.bias):raise ValueError("EF head slice copy failed")
        self.float()
        if sum(p.numel() for p in self.parameters() if p.requires_grad)!=COUNTS[variant]:raise ValueError("Actual unimodal trainable count differs from independent specification")
        if any(n.startswith("fnirs." if self.modality=="eeg" else "eeg.") for n,_ in self.named_parameters()):raise ValueError("Inactive branch registered")

    @property
    def branch(self):return getattr(self,self.modality)

    def forward(self,eeg=None,fnirs=None):
        active=eeg if self.modality=="eeg" else fnirs
        inactive=fnirs if self.modality=="eeg" else eeg
        if active is None or inactive is not None:raise ValueError("Only the active modality may enter forward")
        return self.head(self.dropout(self.branch(active)))

    def load_state_dict(self,state_dict,strict=True,assign=False):
        name=self.modality+".front.bn1.bias"
        if name in state_dict and bool(torch.count_nonzero(state_dict[name])):raise ValueError("Nonzero frozen BN1 bias checkpoint rejected")
        return super().load_state_dict(state_dict,strict=strict,assign=assign)
