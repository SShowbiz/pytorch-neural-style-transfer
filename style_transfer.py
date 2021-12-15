import torchvision.models as models
import pytorch_lightning as pl
import torch
from torch import nn
from utils import gram_matrix, image_loader, next_path
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

MSE = nn.MSELoss()


class Normalization(nn.Module):
    def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):

        # input.clamp_(0, 1)
        # self.target.clamp_(0, 1)

        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)

        # G.clamp_(0, 1)
        # self.target.clamp_(0, 1)

        self.loss = F.mse_loss(G, self.target)
        return input


class StyleTransferNetwork(pl.LightningModule):
    def __init__(self, content_image, style_image, input_image, style_weight, content_weight, image_size):
        super().__init__()

        content_image = image_loader(content_image, image_size)
        style_image = image_loader(style_image, image_size)
        input_image = image_loader(input_image, image_size)

        assert content_image.size() == style_image.size()

        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        model = nn.Sequential(Normalization())
        content_losses = []
        style_losses = []

        VGGNet = models.vgg19(pretrained=True).features.eval()

        index = 0
        for layer in VGGNet.children():
            if isinstance(layer, nn.Conv2d):
                index += 1
                name = f'conv_{index}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{index}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{index}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{index}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module(f'content_loss_{index}', content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f'style_loss_{index}', style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[: (i + 1)]

        self.input_image = input_image
        self.model = model
        self.content_losses = content_losses
        self.style_losses = style_losses
        self.style_weight = style_weight
        self.content_weight = content_weight

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.LBFGS([self.input_image])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)},
        }

    def training_step(self, batch, batch_idx):
        self.input_image.requires_grad_(True)
        self.model.requires_grad_(False)

        self.model(self.input_image)

        with torch.no_grad():
            self.input_image.clamp_(0, 1)

        style_score = sum([style_loss.loss for style_loss in self.style_losses])
        content_score = sum([content_loss.loss for content_loss in self.content_losses])
        loss = style_score * self.style_weight + content_score * self.content_weight

        return loss

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            self.input_image.clamp_(0, 1)
            save_image(self.input_image, next_path('./output_person_images/output_image_%s.jpg'))

    def train_dataloader(self):
        return DataLoader('_')
