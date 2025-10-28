import math
from typing import Dict, Iterable, List, Optional

import torch


class Adam:
    """Lightweight Adam optimizer that operates on raw tensors without autograd."""

    def __init__(
        self,
        param_groups: Iterable[Dict[str, torch.Tensor]],
        betas: Optional[tuple[float, float]] = None,
        eps: float = 1e-8,
    ) -> None:
        if betas is None:
            betas = (0.9, 0.999)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.param_groups: List[Dict[str, torch.Tensor]] = []
        for group in param_groups:
            name = group["name"]
            tensor = group["param"]
            lr = group["lr"]
            if tensor is None:
                raise ValueError(f"Parameter group '{name}' has no tensor reference")
            state = {
                "m": torch.zeros_like(tensor),
                "v": torch.zeros_like(tensor),
            }
            self.param_groups.append(
                {
                    "name": name,
                    "param": tensor,
                    "lr": lr,
                    "state": state,
                }
            )

    def zero_grad(self) -> None:  # noqa: D401 - kept for API symmetry
        """No-op placeholder to mirror torch.optim API."""
        return None

    def step(self, grads: Dict[str, torch.Tensor]) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - math.pow(self.beta1, self.step_count)
        bias_correction2 = 1.0 - math.pow(self.beta2, self.step_count)
        for group in self.param_groups:
            name = group["name"]
            if name not in grads:
                raise KeyError(f"Missing gradient for parameter group '{name}'")
            param = group["param"]
            grad = grads[name]
            if grad is None:
                continue
            if grad.shape != param.shape:
                raise ValueError(
                    f"Gradient shape {grad.shape} does not match parameter shape {param.shape} for '{name}'"
                )
            grad = grad.to(dtype=param.dtype, device=param.device)
            state = group["state"]
            m = state["m"]
            v = state["v"]
            m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            param.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-group["lr"])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "step": self.step_count,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "param_groups": [
                {
                    "name": group["name"],
                    "lr": group["lr"],
                    "state": {
                        "m": group["state"]["m"].clone(),
                        "v": group["state"]["v"].clone(),
                    },
                }
                for group in self.param_groups
            ],
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.step_count = int(state_dict.get("step", 0))
        self.beta1 = float(state_dict.get("beta1", self.beta1))
        self.beta2 = float(state_dict.get("beta2", self.beta2))
        self.eps = float(state_dict.get("eps", self.eps))
        state_groups = {group["name"]: group for group in state_dict.get("param_groups", [])}
        for group in self.param_groups:
            name = group["name"]
            if name not in state_groups:
                continue
            saved = state_groups[name]
            group["lr"] = float(saved.get("lr", group["lr"]))
            saved_state = saved.get("state", {})
            m = saved_state.get("m")
            v = saved_state.get("v")
            if m is not None and v is not None:
                group["state"]["m"].copy_(m)
                group["state"]["v"].copy_(v)

    def rebuild(self, param_groups: Iterable[Dict[str, torch.Tensor]]) -> None:
        """Reset optimizer to track a new set of tensors (used when Gaussians are added)."""
        self.__init__(param_groups, betas=(self.beta1, self.beta2), eps=self.eps)
