import torch.nn as nn
from utils.constants import BackboneNetwork
import torchvision.models as models


class FastFlowerClassifier(nn.Module):


    def __init__(self, num_classes: int, backbone: str):
        self.num_classes = num_classes
        self.backbone = backbone

        self.feature_extractor, in_features = self._load_pretrained_model(backbone)

        # Modify classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )
        self.flatten = nn.Flatten(start_dim=1)


    def _load_pretrained(self):

        assert self.backbone in BackboneNetwork.list(), f"Backbone Network '{backbone}' is not supported."

        if self.backbone == BackboneNetwork.MOBILE_NET:
            model = models.mobilenet_v3_small(pretrained=True)
            feature_extractor = model.features
            in_features = model.classifier[0].in_features

            return feature_extractor, in_features
        if self.backbone == BackboneNetwork.RESNET:
            model = models.resnet18(pretrained=True)
            feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
            in_features = model.fc.in_features

            return feature_extractor, in_features

        if  self.backbone == BackboneNetwork.EFFICIENT_NET:
            model = models.efficientnet_b0(pretrained=True)
            feature_extractor = model.features
            in_features = model.classifier[1].in_features


            return feature_extractor, in_features


    def forward(self, x):
        """
        Forward pass through the feature extractor and classifier.
        """
        x = self.feature_extractor(x)
        x = self.flatten(x)  # Flatten for FC layers
        x = self.classifier(x)
        return x


model = FastFlowerClassifier(num_classes=17, backbone=BackboneNetwork.EFFICIENT_NET).cuda()
print(model)