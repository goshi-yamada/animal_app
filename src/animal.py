# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn

#学習時に使ったのと同じ学習済みモデルをインポートh
from torchvision.models import resnet34




# 学習済みモデルに合わせた前処理を追加 ファインチューニング用に変換
transform = transforms.Compose([
    transforms.Resize(256),#固定
    transforms.CenterCrop(224),
    transforms.ToTensor(),#固定
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),#固定
    transforms.RandomRotation(degrees=[-10, 10])
])

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet34(pretrained=True)
        self.fc = nn.Linear(1000, 10)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h