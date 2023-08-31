# imports
from torch.nn import Identity, ReLU, Linear, Sequential, Sigmoid, Module, Dropout, Softmax

class BallClassifier(Module):
    def __init__(self, base_model, num_classes) -> None:
        print(f'nfeatures {base_model.fc.in_features}')
        super().__init__()
        self.num_classes = num_classes
        self.base_model = base_model
        self.classifier = Sequential(
			Linear(base_model.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 256),
			ReLU(),
			Dropout(),
			Linear(256, self.num_classes),
		)
        self.base_model.fc = Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x













