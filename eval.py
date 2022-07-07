import os
import torch
from lib.dataset import find_dataset_using_name
from lib.models.deciwatch import DeciWatch
from lib.core.evaluate import Evaluator
from torch.utils.data import DataLoader
from lib.utils.utils import prepare_output_dir, worker_init_fn
from lib.core.config import parse_args


def main(cfg):
    dataset_class = find_dataset_using_name(cfg.DATASET_NAME)

    test_dataset = dataset_class(cfg,
                                estimator=cfg.ESTIMATOR,
                                return_type=cfg.BODY_REPRESENTATION,
                                phase='test')

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS_NUM,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)

    model = DeciWatch(test_dataset.input_dimension,
                    sample_interval=cfg.SAMPLE_INTERVAL,
                    encoder_hidden_dim=cfg.MODEL.ENCODER_EMBEDDING_DIMENSION,
                    decoder_hidden_dim=cfg.MODEL.DECODER_EMBEDDING_DIMENSION,
                    dropout=cfg.MODEL.DROPOUT,
                    nheads=cfg.MODEL.ENCODER_HEAD,
                    dim_feedforward=256,
                    enc_layers=cfg.MODEL.ENCODER_TRANSFORMER_BLOCK,
                    dec_layers=cfg.MODEL.ENCODER_TRANSFORMER_BLOCK,
                    activation="leaky_relu",
                    pre_norm=cfg.TRAIN.PRE_NORM,
                    recovernet_interp_method=cfg.MODEL.DECODER_INTERP,
                    recovernet_mode=cfg.MODEL.DECODER).to(cfg.DEVICE)

    if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(
            cfg.EVALUATE.PRETRAINED):
        checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
        performance = checkpoint['performance']
        model.load_state_dict(checkpoint['state_dict'])
        print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}...')
    else:
        print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
        exit()

    evaluator = Evaluator(model=model, test_loader=test_loader, cfg=cfg)
    evaluator.calculate_flops()
    evaluator.calculate_parameter_number()
    evaluator.run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)