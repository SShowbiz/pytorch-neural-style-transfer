import pytorch_lightning as pl
import argparse
import pathlib
from style_transfer import StyleTransferNetwork


def train(args):
    STNet = StyleTransferNetwork(**vars(args))

    trainer = pl.Trainer(max_steps=300, min_steps=300, precision=32)
    trainer.fit(STNet, train_dataloaders=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_image', type=pathlib.Path, required=True)
    parser.add_argument('--style_image', type=pathlib.Path, required=True)
    parser.add_argument('--input_image', type=pathlib.Path, required=True)
    parser.add_argument('--style_weight', type=int, default=1000000, required=False)
    parser.add_argument('--content_weight', type=int, default=1, required=False)
    parser.add_argument('--image_size', type=int, default=1, required=False)

    args = parser.parse_args()
    train(args)
