import torch
from torch import nn
from .student import Student


class Teacher(nn.Module):
    """Frozen adapter. Supply another RGB-output nn.Module to replace the backbone."""
    def __init__(self, model):
        super().__init__()
        self.model = model.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)


def build_teacher(config, device):
    mode = config['mode']
    if mode in ('none', 'oracle'):
        return None
    if mode != 'checkpoint' or not config.get('checkpoint'):
        raise ValueError('Teacher requires mode=checkpoint and a pretrained checkpoint path')
    model = Student(config.get('width', 48))
    state = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('student', state), strict=True)
    return Teacher(model).to(device)
