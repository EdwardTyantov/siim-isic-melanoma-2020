import logging
from collections import defaultdict


class GradientEstimator(object):
    """Gradients estimator for logging during training."""
    def __init__(self, model, param_groups):
        self._model = model
        self._param_groups = param_groups

    def __call__(self, losses):
        metrics = defaultdict(dict)
        self._model.zero_grad()
        for loss_name, loss in losses.items():
            try:
                loss.backward(retain_graph=True)
            except RuntimeError:
                logging.error("Error occured during {} gradient estimation".format(loss_name))
                raise
            for layer_name, grad_value in self._get_gradients(self._param_groups).items():
                metrics[f'GradL1_{loss_name}_{layer_name}'] = grad_value
            self._model.zero_grad()
        return metrics

    def _get_gradients(self, param_groups):
        grads = {}
        for i, param_group in enumerate(param_groups):
            grad_sum, n = 0, 0
            for parameter in param_group['params']:
                if parameter.grad is not None:
                    mean_grad = parameter.grad.abs().mean().item()
                    grad_sum = grad_sum + mean_grad
                    n += 1
            if n > 0:
                grads[f'layer_{i}'] = grad_sum / n
        return grads
