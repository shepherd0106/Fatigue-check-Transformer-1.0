"""Core seven real architectures; import this module only with a torch runtime.

Mamba is the official locked-source implementation, never an approximate substitute.
The dedicated single-writer development process freezes torch-conv policy on first import.
"""
import contextlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from .development import COUNTS, MODELS, bin_boundaries, configuration, stream_seed
from .models import Frontend

_BACKEND = None


def official_mamba():
    global _BACKEND
    import importlib.metadata
    import mamba_ssm.modules.mamba_simple as module
    version = importlib.metadata.version("mamba-ssm")
    if version != "2.2.6.post3":
        raise ValueError(f"Locked mamba-ssm 2.2.6.post3 required, found {version}")
    # Frozen once in this dedicated process. Never toggle between fits or threads.
    if _BACKEND is None:
        _BACKEND = dict(policy="torch_conv_selective_scan_cuda",version=version,
                        original_causal_conv_available=module.causal_conv1d_fn is not None)
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = None
    if module.causal_conv1d_fn is not None or module.causal_conv1d_update is not None:
        raise ValueError("Mamba backend bindings changed after process startup")
    return module.Mamba


def backend_policy():
    official_mamba()
    return dict(_BACKEND)


@contextlib.contextmanager
def scoped_rng(seed, device=None):
    devices=[] if device is None or torch.device(device).type!="cuda" else [torch.device(device).index or 0]
    with torch.random.fork_rng(devices=devices):
        # Avoid manual_seed affecting CUDA devices outside the saved scope.
        torch.random.default_generator.manual_seed(int(seed))
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(int(seed))
        yield


class StreamDropout(nn.Module):
    """Each dropout site has a saved counter and a stable isolated RNG namespace."""
    def __init__(self, seed, p=.3):
        super().__init__(); self.p=p; self.seed=int(seed)
        self.register_buffer("calls",torch.zeros((),dtype=torch.int64))

    def forward(self,x):
        if not self.training: return x
        counter=int(self.calls.item())
        with scoped_rng((self.seed+counter)%(2**63-1),x.device):
            out=F.dropout(x,p=self.p,training=True)
        self.calls.add_(1)
        return out


class MambaLayer(nn.Module):
    def __init__(self,seed):
        super().__init__()
        self.norm=nn.LayerNorm(32,eps=1e-5)
        cls=official_mamba()
        self.encoder=cls(d_model=32,d_state=16,d_conv=4,expand=2,dt_rank=2,
            bias=False,conv_bias=True,dt_min=.001,dt_max=.1,dt_init="random",
            dt_scale=1,dt_init_floor=1e-4,use_fast_path=False)
        self.dropout=StreamDropout(seed)

    def forward(self,z):
        official_mamba()  # fail if global policy was changed
        return z+self.dropout(self.encoder(self.norm(z),inference_params=None))


class TransformerLayer(nn.Module):
    def __init__(self,seed):
        super().__init__();self.seed=int(seed)
        self.register_buffer("calls",torch.zeros((),dtype=torch.int64))
        self.encoder=nn.TransformerEncoderLayer(32,4,64,dropout=.3,activation="gelu",
            layer_norm_eps=1e-5,batch_first=True,norm_first=True,bias=True)

    def forward(self,z):
        # Attention's functional dropout also lives in this layer-specific stream.
        if not self.training: return self.encoder(z)
        with scoped_rng((self.seed+int(self.calls.item()))%(2**63-1),z.device):
            out=self.encoder(z)
        self.calls.add_(1)
        return out


class Readout(nn.Module):
    def __init__(self,segment):
        super().__init__();self.norm=nn.LayerNorm(32,eps=1e-5)
        self.segment=nn.Linear(128,32) if segment else None
        if self.segment is not None:
            nn.init.zeros_(self.segment.weight);nn.init.zeros_(self.segment.bias)

    def forward(self,z):
        g=z.mean(dim=1)
        if self.segment is not None:
            bounds=bin_boundaries(z.shape[1])
            p=torch.cat([z[:,bounds[j]:bounds[j+1]].mean(dim=1) for j in range(4)],dim=1)
            g=g+self.segment(p)
        return self.norm(g)


class Branch(nn.Module):
    def __init__(self,name,mode,segment,seed,fold):
        super().__init__();cfg=configuration()[name]; self.mode=mode
        with scoped_rng(stream_seed(seed,fold,f"base.{name}.frontend")):
            self.front=Frontend(cfg)
        # bn1's channel shift passes through a linear spatial convolution and is
        # removed by training-mode bn2 centering. Keep its checkpoint key, but
        # remove this redundant trainable freedom rather than masking gradients.
        nn.init.zeros_(self.front.bn1.bias)
        self.front.bn1.bias.requires_grad_(False)
        self.front.dropout1=StreamDropout(stream_seed(seed,fold,f"dropout.{name}.pool1"))
        self.front.dropout2=StreamDropout(stream_seed(seed,fold,f"dropout.{name}.pool2"))
        self.layers=nn.ModuleList()
        for j in range(2 if mode!="cnn" else 0):
            with scoped_rng(stream_seed(seed,fold,f"base.{name}.{mode}.layer{j}")):
                layer=(MambaLayer if mode=="mamba" else TransformerLayer)(stream_seed(seed,fold,f"dropout.{name}.{mode}.layer{j}"))
            self.layers.append(layer)
        with scoped_rng(stream_seed(seed,fold,f"readout.{name}")):
            self.readout=Readout(segment)
        pe=torch.zeros(1,cfg["tokens"],32,dtype=torch.float32)
        if mode=="transformer":
            position=torch.arange(cfg["tokens"],dtype=torch.float32)[:,None]
            div=torch.exp(torch.arange(0,32,2,dtype=torch.float32)*(-math.log(10000)/32))
            pe[0,:,0::2]=torch.sin(position*div);pe[0,:,1::2]=torch.cos(position*div)
        self.register_buffer("pe",pe)

    def forward(self,x):
        z=self.front(x)
        if tuple(z.shape[1:])!=tuple(self.pe.shape[1:]):raise ValueError("Unexpected tokens")
        if self.mode=="transformer":z=z+self.pe
        for layer in self.layers:z=layer(z)
        return self.readout(z)


class DevelopmentModel(nn.Module):
    def __init__(self,variant,seed=17,fold=2):
        super().__init__()
        if variant not in MODELS:raise ValueError("Unknown development model")
        self.variant=variant
        mode="cnn" if variant=="C_EF" else "transformer" if variant=="T_EF" else "mamba"
        segment=variant in ("Q_EF","F_EF")
        self.eeg=Branch("eeg",mode,segment,seed,fold)
        self.fnirs=Branch("fnirs",mode,segment,seed,fold)
        self.condition=None;self.capacity=None
        if variant in ("A_EF","F_EF"):
            with scoped_rng(stream_seed(seed,fold,"fusion.condition")):
                self.condition=nn.Sequential(nn.Linear(32,32),nn.GELU(approximate="none"),nn.Linear(32,64))
            nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
        if variant=="K_EF":
            with scoped_rng(stream_seed(seed,fold,"fusion.capacity")):
                self.capacity=nn.Sequential(nn.Linear(64,88),nn.GELU(approximate="none"),nn.Linear(88,64))
            nn.init.zeros_(self.capacity[-1].weight);nn.init.zeros_(self.capacity[-1].bias)
        self.dropout=StreamDropout(stream_seed(seed,fold,"dropout.head"))
        with scoped_rng(stream_seed(seed,fold,"base.head")):
            self.head=nn.Linear(64,2)
        self.float()
        count=sum(p.numel() for p in self.parameters() if p.requires_grad)
        if count!=COUNTS[variant]:raise ValueError(f"Parameter mismatch {variant}: {count} != {COUNTS[variant]}")

    def load_state_dict(self,state_dict,strict=True,assign=False):
        for name in ("eeg.front.bn1.bias","fnirs.front.bn1.bias"):
            if name in state_dict and bool(torch.count_nonzero(state_dict[name])):
                raise ValueError("Nonzero legacy bn1 bias is incompatible with the frozen-zero model")
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def forward(self,eeg,fnirs,return_modulation=False):
        e,f=self.eeg(eeg),self.fnirs(fnirs)
        diagnostics=None
        if self.condition is not None:
            a,b=self.condition(f).chunk(2,dim=1)
            gamma,beta=.1*torch.tanh(a),.1*torch.tanh(b)
            e=(1+gamma)*e+beta
            if return_modulation:
                diagnostics=dict(gamma=gamma.detach(),beta=beta.detach(),
                                 saturation=((torch.abs(gamma)>=.099)|(torch.abs(beta)>=.099)).detach())
        x=torch.cat([e,f],dim=1)
        if self.capacity is not None:x=x+self.capacity(x)
        logits=self.head(self.dropout(x))
        return (logits,diagnostics) if return_modulation else logits


def optimizer_groups(model):
    decay=[];no_decay=[];names={"decay":[],"no_decay":[]};seen=set()
    norms=set()
    for prefix,module in model.named_modules():
        if isinstance(module,(nn.LayerNorm,nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
            norms.update((prefix+"." if prefix else "")+n for n,_ in module.named_parameters(recurse=False))
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if id(p) in seen:raise ValueError("Duplicate optimizer parameter")
        seen.add(id(p))
        exempt=name.endswith(".bias") or name in norms or p.ndim<=1 or getattr(p,"_no_weight_decay",False) or name.endswith((".A_log",".D"))
        group="no_decay" if exempt else "decay"
        (no_decay if exempt else decay).append(p);names[group].append(name)
    expected={id(p) for p in model.parameters() if p.requires_grad}
    if seen!=expected or set(names["decay"])&set(names["no_decay"]):raise ValueError("Optimizer coverage failure")
    return [dict(params=decay,weight_decay=.001),dict(params=no_decay,weight_decay=0.)],names
