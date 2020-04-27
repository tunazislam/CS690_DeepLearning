import torch


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten()
        )

        # Random initialization of weights
        torch.nn.init.kaiming_normal_(self.gamma[0].weight)
        torch.nn.init.zeros_(self.gamma[0].bias)
        torch.nn.init.kaiming_normal_(self.gamma[3].weight)
        torch.nn.init.zeros_(self.gamma[3].bias)

    def forward(self, x):
        return self.gamma(x)

    def get_repr_size(self, input_sz=32):
        sz_repr = (input_sz - 5 + 1) // 2
        sz_repr = (sz_repr - 5 + 1) // 2
        sz_repr = sz_repr * sz_repr * 16
        return sz_repr

class CNN(torch.nn.Module):
    def __init__(self, n_hidden=100, n_classes=10):
        super().__init__()

        self.representation = FeatureExtractor()
        in_features = self.representation.get_repr_size()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_classes)
        )

        # Random initialization of weights
        torch.nn.init.kaiming_normal_(self.classifier[0].weight)
        torch.nn.init.zeros_(self.classifier[0].bias)
        torch.nn.init.kaiming_normal_(self.classifier[2].weight)
        torch.nn.init.zeros_(self.classifier[2].bias)

    def forward(self, X, return_representation=False):
        z = self.representation(X)
        h = self.classifier(z)
        if return_representation:
            return h, z
        else:
            return h