"""Paired full-model Adam interventions. No gradient-through-optimizer claims."""
import copy
import torch
from ..models.knowledge_views import KnowledgeViews
from ..losses import ReconstructionLoss

# Nonempty subsets; each action keeps the same total KD coefficient.
ACTIONS = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0],
                        [1,0,1], [0,1,1], [1,1,1]], dtype=torch.float32)
ACTIONS /= ACTIONS.sum(1, keepdim=True)


class RestorationResponse:
    def __init__(self, ssim_weight=.2):
        self.rec = ReconstructionLoss(ssim_weight)
        self.views = KnowledgeViews()

    def __call__(self, prediction, target):
        return torch.cat([self.rec(prediction, target).reshape(1),
                          self.views(prediction, target).mean(0)])


class ShadowAdamEvaluator:
    def __init__(self, kd_weight=.1, ssim_weight=.2):
        self.kd_weight = kd_weight
        self.response = RestorationResponse(ssim_weight)

    def step_copy(self, student, optimizer, x, target, teacher_output, action=None, candidate=0):
        if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
            raise ValueError('Shadow evaluator currently supports single-group torch.optim.Adam')
        devices = [x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
        # Every branch starts with the same torch RNG; real training state is untouched.
        with torch.random.fork_rng(devices=devices):
            model = copy.deepcopy(student)
            virtual = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
            virtual.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            model.train()
            output = model(x)
            loss = self.response.rec(output, target)
            if action is not None:
                losses = self.response.views(output[candidate:candidate+1], teacher_output[candidate:candidate+1])
                loss = loss + self.kd_weight*(losses*action.to(x)).sum()/len(x)
            virtual.zero_grad(set_to_none=True)
            loss.backward()
            virtual.step()
            return model.eval()

    def measure(self, student, optimizer, x, target, teacher_output, references, candidate=0):
        base = self.step_copy(student, optimizer, x, target, teacher_output)
        with torch.no_grad():
            baseline = torch.stack([self.response(base(rx), rt) for rx, rt in references])
        del base
        responses = []
        for action in ACTIONS:
            branch = self.step_copy(student, optimizer, x, target, teacher_output, action, candidate)
            with torch.no_grad():
                changed = torch.stack([self.response(branch(rx), rt) for rx, rt in references])
                responses.append(baseline-changed)
            del branch
        return torch.stack(responses).detach(), baseline.detach()
