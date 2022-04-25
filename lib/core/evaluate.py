import torch
from lib.utils.eval_metrics import *
from lib.models.smpl import SMPL
from lib.utils.geometry_utils import *
from thop import profile


class Evaluator():

    def __init__(
        self,
        test_loader,
        model,
        cfg,
    ):
        self.test_dataloader = test_loader
        self.model = model
        self.device = cfg.DEVICE
        self.cfg = cfg

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self):
        self.evaluate()

    def calculate_parameter_number(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters()
                            if p.requires_grad)
        log_str = f'Total Parameters: {total_num/(1000 ** 2)} M, Trainable Parameters: {trainable_num /(1000 ** 2)} M'
        print(log_str)
        return {'Total': total_num, 'Trainable': trainable_num}

    def calculate_flops(self):
        data = torch.randn(
            (1, self.cfg.MODEL.SLIDE_WINDOW_SIZE,
             self.model.deciwatch_par["input_dim"])).to(self.device)
        flops, _ = profile(self.model, inputs=(data, self.device))
        log_str = f'Flops Per Frame: {flops/self.cfg.MODEL.SLIDE_WINDOW_SIZE/(1000 ** 3)} G'
        print(log_str)
        return {'Flops': flops}

    def evaluate_3d(self):
        eval_dict = evaluate_deciwatch_3D(self.model, self.test_dataloader,
                                          self.device, self.cfg)

        log_str = ' '.join(
            [f'{k.upper()}: {v:.2f},' for k, v in eval_dict.items()])
        print(log_str)

        return eval_dict

    def evaluate_smpl(self):

        eval_dict = evaluate_deciwatch_smpl(self.model, self.test_dataloader,
                                            self.device, self.cfg)

        log_str = ' '.join(
            [f'{k.upper()}: {v:.2f},' for k, v in eval_dict.items()])
        print(log_str)

        return eval_dict

    def evaluate_2d(self):
        eval_dict = evaluate_deciwatch_2D(self.model, self.test_dataloader,
                                          self.device, self.cfg)

        log_str = "" + ' '.join(
            [f'{k.upper()}: {v*100:.2f}%,' for k, v in eval_dict.items()])
        print(log_str)

        return eval_dict

    def evaluate(self):
        self.model.eval()
        if self.cfg.BODY_REPRESENTATION == "3D":
            return self.evaluate_3d()

        elif self.cfg.BODY_REPRESENTATION == "smpl":
            return self.evaluate_smpl()

        elif self.cfg.BODY_REPRESENTATION == "2D":
            return self.evaluate_2d()
