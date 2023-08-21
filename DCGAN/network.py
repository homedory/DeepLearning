import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=(4,4), stride=(1,1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.upconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.Tanh()
        )

        nn.init.xavier_uniform_(self.upconv1.weight)
        nn.init.xavier_uniform_(self.upconv2.weight)
        nn.init.xavier_uniform_(self.upconv3.weight)
        nn.init.xavier_uniform_(self.upconv4.weight)
        nn.init.xavier_uniform_(self.upconv5.weight)

    def forward(self, x):

        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.Sigmoid()
        )

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x.squeeze()
