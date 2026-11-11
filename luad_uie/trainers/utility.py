import torch
from torch.func import functional_call


def _flatten(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def estimate_utility(student, x, teacher_output, eval_x, eval_target, views, reconstruction, eta=1e-3, virtual=False):
    """Per-candidate labels on disjoint TRAIN-INTERNAL eval data.

    Head-only dot products approximate a head-only SGD step. No optimizer or
    parameter mutations; cloned buffers also protect replacement backbones.
    """
    was_training = student.training
    student.eval()
    try:
        parameters = student.utility_parameters()
        selected = tuple(parameters.values())
        rec = reconstruction(student(eval_x), eval_target)
        gr = _flatten(torch.autograd.grad(rec, selected)).detach()
        losses = views(student(x), teacher_output).mean(0)
        dots, cosines, norms, gains = [], [], [], []
        for k in range(3):
            grads = torch.autograd.grad(losses[k], selected, retain_graph=k < 2)
            gk = _flatten(grads).detach()
            dot = torch.dot(gr, gk)
            dots.append(dot)
            norms.append(gk.norm())
            cosines.append(dot/(gr.norm()*gk.norm()).clamp_min(1e-12))
            if virtual:
                params = {name: p.detach().clone() for name, p in student.named_parameters()}
                for (name, _), grad in zip(parameters.items(), grads):
                    params[name] -= eta*grad.detach()
                buffers = {name: b.detach().clone() for name, b in student.named_buffers()}
                with torch.no_grad():
                    changed = functional_call(student, (params, buffers), (eval_x,))
                    gains.append(rec.detach()-reconstruction(changed, eval_target))
        result = {'dot': torch.stack(dots), 'cosine': torch.stack(cosines),
                  'kd_grad_norm': torch.stack(norms), 'rec_grad_norm': gr.norm()}
        if virtual:
            result['virtual_gain'] = torch.stack(gains)
        return {k: v.detach() for k, v in result.items()}
    finally:
        student.train(was_training)
