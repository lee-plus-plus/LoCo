import torch


def _get_model_device(model):
    return next(model.parameters()).device


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """usage:

    >> model = create_model().cuda()
    >> model_ema = ExponentialMovingAverage(model, decay=0.99, device='cuda')

    >> train_epoch(model, train_loader, criterion)
    >> model_ema.update_parameters(model)

    >> eval_epoch(model, valid_loader, criterion)
    >> eval_epoch(model_ema, valid_loader, criterion)
    """

    def __init__(self, model, decay, device=None):
        device = device or _get_model_device(model)

        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)
        # if self.n_averaged == 0, it will regard the next update_parameters()
        # as a reset instead of an update, which disobey our intuition
        self.n_averaged += 1
