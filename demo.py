import os
import torch
from lib.dataset import find_dataset_using_name
from lib.models.deciwatch import DeciWatch
from lib.core.config import parse_args
from lib.visualize.visualize import Visualize


def main(cfg):

    dataset_class = find_dataset_using_name(cfg.DATASET_NAME)
    test_dataset = dataset_class(cfg,
                                estimator=cfg.ESTIMATOR,
                                return_type=cfg.BODY_REPRESENTATION,
                                phase='test')

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

    visualizer = Visualize(test_dataset,cfg)

    if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(
            cfg.EVALUATE.PRETRAINED):
        checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}...')
    else:
        print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
        exit()
    
    visualizer.visualize(model)


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    main(cfg)