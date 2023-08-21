import torch.nn as nn
import torch
import math


class ConditioningAugmention(nn.Module):
    def __init__(self, input_dim, emb_dim, device):
        super(ConditioningAugmention, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        # [FC + Activation] x1

        self.layer = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Inputs:
            x: CLIP text embedding c_txt
        Outputs:
            condition: augmented text embedding \hat{c}_txt
            mu: mean of x extracted from self.layer. Length : half of output from self.layer
            log_sigma: log(sigma) of x extracted from self.layer. Length : half of output from self.layer

        calculate: condition = mu + exp(log_sigma) * z, z ~ N(0, 1)
        '''

        x = self.layer(x)
        mu, log_sigma = x.chunk(2, dim=1)

        z = torch.randn_like(mu, device=self.device)
        condition = mu + torch.exp(log_sigma) * z

        return condition, mu, log_sigma


class ImageExtractor(nn.Module):
    def __init__(self, in_chans):
        super(ImageExtractor, self).__init__()
        self.in_chans = in_chans
        self.out_chans = 3

        #  [TransposeConv2d + Activation] x1

        self.image_net = nn.Sequential(
            nn.Conv2d(self.in_chans, self.out_chans, 3, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, x):

        '''
        Inputs:
            x: input tensor, shape [C, H, W]
        Outputs:
            out: output image extracted with self.image_net, shape [3, H, W]

        TODO: calculate out
        '''

        out = self.image_net(x)

        return out


class Generator_type_1(nn.Module):
    def __init__(self, in_chans, input_dim):
        super(Generator_type_1, self).__init__()
        self.in_chans = in_chans
        self.input_dim = input_dim

        self.mapping = self._mapping_network()
        self.upsample_layer = self._upsample_network()
        self.image_net = self._image_net()

    def _image_net(self):
        return ImageExtractor(self.in_chans // 16)

    def _mapping_network(self):

        # [FC + BN + LeakyReLU] x1
        # Change the input tensor dimension [projection_dim + noise_dim] into [Ng * 4 * 4]

        return nn.Sequential(
            nn.Linear(self.input_dim, self.in_chans * 4 * 4),
            nn.BatchNorm1d(self.in_chans * 4 * 4),  # First reshape and then BatchNorm2d?
            nn.LeakyReLU()
        )

    def _upsample_network(self):

        # [ConvTranspose2D + BN + ReLU] x4
        # Change the input tensor dimension [Ng, 4, 4] into [Ng/16, 64, 64]
        layer = []
        in_channels = self.in_chans
        for _ in range(4):
            layer.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, int(in_channels / 2), 4, stride=2, padding=1),
                    nn.BatchNorm2d(int(in_channels / 2)),
                    nn.ReLU()
                )
            )
            in_channels = int(in_channels / 2)
        return nn.Sequential(*layer)

    def forward(self, condition, noise):
        '''
        Inputs:
            condition: \hat{c}_txt, shape [projection_dim]
            noise: gaussian noise sampled from N(0, 1), shape [noise_dim]
        Outputs:
            out: tensor extracted from self.upsample_layer, shape [Ng/16, 64, 64]
            out_image: image extracted from self.image_net, shape [3, 64, 64]
        '''
        out = torch.cat((condition, noise), 1)
        out = self.mapping(out)
        out = out.reshape(-1, self.in_chans, 4, 4)
        out = self.upsample_layer(out)

        out_image = self.image_net(out)

        return out, out_image

class Generator_type_2(nn.Module):
    def __init__(self, in_chans, condition_dim, num_res_layer, device):
        super(Generator_type_2, self).__init__()
        self.device = device

        self.in_chans = in_chans
        self.condition_dim = condition_dim
        self.num_res_layer = num_res_layer

        self.joining_layer = self._joint_conv()
        self.res_layer = nn.ModuleList(
            [self._res_layer() for _ in range(self.num_res_layer)])
        self.upsample_layer = self._upsample_network()
        self.image_net = self._image_net()

    def _image_net(self):
        return ImageExtractor(self.in_chans // 2)

    def _upsample_network(self):
        # [ConvTranspose2D + BN + ReLU] x1
        # Change the input tensor dimension [C, H, W] into [C/2, 2H, 2W]

        return nn.Sequential(
            nn.ConvTranspose2d(self.in_chans, int(self.in_chans / 2), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(self.in_chans / 2)),
            nn.ReLU()
        )


    def _joint_conv(self):

        # [Conv2d + BN + ReLU] x1
        # change the channel size of input tensor into self.in_chans

        return nn.Sequential(
            nn.Conv2d(self.condition_dim + self.in_chans, self.in_chans, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_chans),
            nn.ReLU()
        )

    def _res_layer(self):
        return ResModule(self.in_chans)

    def forward(self, condition, prev_output):
        '''
        Inputs:
            condition: \hat{c}_txt, shape [projection_dim]
            prev_output: 'out' tensor returned from previous stage generator, shape [C, H, W]
        Outputs:
            out: tensor extracted from self.upsample_layer, shape [C/2, 2H, 2W]
            out_image: image extracted from self.image_net, shape [3, 2H, 2W]
        '''
        res = prev_output.shape[-1]  # spatial size

        condition = condition.reshape(-1, condition.shape[1], 1, 1)
        condition = condition.repeat(1, 1, res, res)
        out = torch.cat((condition, prev_output), 1)
        out = self.joining_layer(out)

        for layer in self.res_layer:
            out = layer(out)

        out = self.upsample_layer(out)
        out_image = self.image_net(out)

        return out, out_image


class ResModule(nn.Module):
    def __init__(self, in_chans):
        super(ResModule, self).__init__()
        self.in_chans = in_chans

        # [Conv2d + BN] + ReLU + [Conv2d + BN]

        self.layer = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 3, 1, 1),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(),
            nn.Conv2d(in_chans, in_chans, 3, 1, 1),
            nn.BatchNorm2d(in_chans)
        )

    def forward(self, x):
        '''
        Inputs:
            x: input tensor, shape [C, H, W]
        Outputs:
            res_out: output tensor, shape [C, H, W]
        TODO: implement residual connection
        '''

        res_out = x + self.layer(x)

        return res_out


class Generator(nn.Module):
    def __init__(self, text_embedding_dim, projection_dim, noise_input_dim, in_chans, out_chans, num_stage, device):
        super(Generator, self).__init__()
        self.device = device

        self.text_embedding_dim = text_embedding_dim
        self.condition_dim = projection_dim
        self.noise_dim = noise_input_dim
        self.input_dim = self.condition_dim + self.noise_dim
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.num_stage = num_stage
        self.num_res_layer_type2 = 2

        # return layers
        self.condition_aug = self._conditioning_augmentation()
        self.g_layer = nn.ModuleList(
            [self._stage_generator(i) for i in range(self.num_stage)])

    def _conditioning_augmentation(self):

        return ConditioningAugmention(self.text_embedding_dim, self.condition_dim, self.device)

    def _stage_generator(self, i):

        if i == 0:
            return Generator_type_1(self.in_chans, self.input_dim)
        else:
            return Generator_type_2(int(self.in_chans / (16 * 2 ** (i - 1))), self.condition_dim,
                                    self.num_res_layer_type2, self.device)

    def forward(self, text_embedding, noise):
        '''
        Inputs:
            text_embedding: c_txt
            z: gaussian noise sampled from N(0, 1)
        Outputs:
            fake_images: List that containing the all fake images generated from each stage's Generator
            mu: mean of c_txt extracted from CANet
            log_sigma: log(sigma) of c_txt extracted from CANet
        '''

        condition, mu, log_sigma = self.condition_aug(text_embedding)

        fake_images = []
        for layer in self.g_layer:
            noise, fake_image = layer(condition, noise)
            fake_images.append(fake_image)

        return fake_images, mu, log_sigma


class UncondDiscriminator(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UncondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans


        # [Conv2d]
        # Change the input tensor dimension [8Nd, 4, 4] into [1, 1, 1]

        self.uncond_layer = nn.Conv2d(8 * self.in_chans, self.out_chans, 4, stride=1, padding=0)

    def forward(self, x):
        '''
        Inputs:
            x: input tensor extracted from prior layer, shape [8Nd, 4, 4]
        Outputs:
            uncond_out: output tensor extracted frm self.uncond_layer, shape [1, 1, 1]
        '''
        uncond_out = self.uncond_layer(x)

        return uncond_out


class CondDiscriminator(nn.Module):
    def __init__(self, in_chans, condition_dim, out_chans):
        super(CondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.condition_dim = condition_dim
        self.out_chans = out_chans

        # [Conv2d + BN + LeakyReLU] + Conv2d
        # Change the input tensor dimension [8Nd + projection_dim, 4, 4] into [1, 1, 1]

        self.cond_layer = nn.Sequential(
            nn.Conv2d(8 * self.in_chans + self.condition_dim, self.out_chans, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_chans),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_chans, 1, 4, stride=1, padding=0),
        )

    def forward(self, x, c):
        '''
        Inputs:
            x: input tensor extracted from prior layer, shape [8Nd, 4, 4]
            c: mu extracted from CANet, shape [projection_dim]
        Outputs:
            cond_out: output tensor extracted frm self.cond_layer, shape [1, 1, 1]
        '''
        assert len(c.shape) == 2, "Condition vector shape mismatched"
        c = c.view(-1, c.shape[-1], 1, 1)
        c = c.repeat(1, 1, 4, 4)

        x = torch.cat((x, c), 1)

        cond_out = self.cond_layer(x)

        return cond_out


class AlignCondDiscriminator(nn.Module):
    def __init__(self, in_chans, condition_dim, text_embedding_dim):
        super(AlignCondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.condition_dim = condition_dim
        self.text_embedding_dim = text_embedding_dim

        # [Conv2d + BN + SiLU] + Conv2d
        # Change the input tensor dimension [8Nd + projection_dim, 4, 4] into [1, 1, 1]

        self.align_layer = nn.Sequential(
            nn.Conv2d(8 * self.in_chans + self.condition_dim, 8 * self.in_chans, 1, stride=1, padding=0),
            nn.BatchNorm2d(8 * self.in_chans),
            nn.SiLU(),
            nn.Conv2d(8 * self.in_chans, self.text_embedding_dim, 4, stride=1, padding=0),
        )

    def forward(self, x, c):
        '''
        Inputs:
            x: input tensor extracted from prior layer, shape [8Nd, 4, 4]
            c: mu extracted from CANet, shape [projection_dim]
        Outputs:
            align_out: output tensor extracted frm self.align_layer, shape [clip_embedding_dim, 1, 1]
        '''

        c = c.view(-1, c.shape[-1], 1, 1)
        c = c.repeat(1, 1, 4, 4)

        x = torch.cat((x, c), 1)

        align_out = self.align_layer(x)
        align_out = align_out.squeeze()

        return align_out


class Discriminator(nn.Module):
    def __init__(self, projection_dim, img_chans, in_chans, out_chans, text_embedding_dim, curr_stage, device):
        super(Discriminator, self).__init__()
        self.condition_dim = projection_dim
        self.img_chans = img_chans
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.text_embedding_dim = text_embedding_dim
        self.curr_stage = curr_stage
        self.device = device

        self.global_layer = self._global_discriminator()
        self.prior_layer = self._prior_layer()
        self.uncond_discriminator = self._uncond_discriminator()
        self.cond_discriminator = self._cond_discriminator()
        self.align_cond_discriminator = self._align_cond_discriminator()

    def _global_discriminator(self):

        # [Conv2d + LeakyReLU] + [Conv2d + BN + LeakyReLU] x 3
        # Change the input tensor dimension [3, H, W] into [8Nd, H/16, W/16]

        return nn.Sequential(
            nn.Conv2d(self.img_chans, self.in_chans, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.in_chans, 2 * self.in_chans, 3, 2, 1),
            nn.BatchNorm2d(2 * self.in_chans),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.in_chans, 4 * self.in_chans, 3, 2, 1),
            nn.BatchNorm2d(4 * self.in_chans),
            nn.LeakyReLU(),
            nn.Conv2d(4 * self.in_chans, 8 * self.in_chans, 3, 2, 1),
            nn.BatchNorm2d(8 * self.in_chans),
            nn.LeakyReLU(),
        )

    def _prior_layer(self):

        # Change the input tensor dimension [8Nd, H/16, W/16] into [8Nd, 4, 4]

        if self.curr_stage == 0:
            return nn.Identity()

        # Stage 0: 64x64, Stage 1: 128x128, Stage 2: 256x256, ...
        layers = []
        for i in range(self.curr_stage):
            layers.append(nn.Sequential(
                                nn.Conv2d(8 * self.in_chans, 2 * 8 * self.in_chans, 3, stride=2, padding=1),
                                nn.BatchNorm2d(2* 8 * self.in_chans),
                                nn.LeakyReLU(),

                                nn.Conv2d(2 * 8 * self.in_chans, 8 * self.in_chans, 3, stride=1, padding=1),
                                nn.BatchNorm2d(8 * self.in_chans),
                                nn.LeakyReLU()
                                )
            )

        return nn.Sequential(*layers)

    def _uncond_discriminator(self):
        return UncondDiscriminator(self.in_chans, self.out_chans)

    def _cond_discriminator(self):
        return CondDiscriminator(self.in_chans, self.condition_dim, self.out_chans)

    def _align_cond_discriminator(self):
        return AlignCondDiscriminator(self.in_chans, self.condition_dim, self.text_embedding_dim)

    def forward(self,
                img,
                condition=None,  # for conditional loss (mu)
                ):

        '''
        Inputs:
            img: fake/real image, shape [3, H, W]
            condition: mu extracted from CANet, shape [projection_dim]
        Outputs:
            out: fake/real prediction result (common output of discriminator)
            align_out: f_real/f_fake extracted from self.align_cond_discriminator for contrastive learning
        '''
        prior = self.global_layer(img)
        prior = self.prior_layer(prior)
        if condition is None:
            out = self.uncond_discriminator(prior)
            align_out = None
        else:
            out = self.cond_discriminator(prior, condition)
            align_out = self.align_cond_discriminator(prior, condition)

        out = out.view(out.shape[0], -1)
        out = nn.Sigmoid()(out)

        return out, align_out


def weight_init(layer):

    if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)

    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias.data, val=0)

    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, val=0.0)
