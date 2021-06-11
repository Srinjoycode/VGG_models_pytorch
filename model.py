'''
    An implementaiton of VGG-16 CNN architechture
'''
# Import the needed libraries
import torch
import torch.nn as nn
import torchvision
from torch import optim  # For optimizers like SGD, Adam, etc.
from tqdm import tqdm  # For nice progress bar!
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# defining the VGG-16 model
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M", ],
}


class VGG_model(nn.Module):
    def __init__(self, in_channels=in_channels, num_classes=num_classes):
        super(VGG_model, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])
        self.fcs = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)



model = VGG_model(in_channels=in_channels, num_classes=num_classes).to(device=device)
print(model)


# Loading the imageNet datasets
# Data augmentation and normalization for training
# Just normalization for validation

tranform = torchvision.transforms.Compose([transforms.ToTensor()])

train_ds = torchvision.datasets.CIFAR10(root='./data',
                                      download=True,
                                      train=True,
                                      transform=tranform,
                                      )
test_ds = torchvision.datasets.CIFAR10(root='./data',
                                     download=True,
                                     train=False,
                                     transform=tranform,
                                     )
train_data_loader = torch.utils.data.DataLoader(train_ds,
                                                batch_size=batch_size,
                                                shuffle=True,

                                                )

test_data_loader = torch.utils.data.DataLoader(test_ds,
                                               batch_size=batch_size,
                                               shuffle=True)
train_features, train_labels = next(iter(train_data_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# training the model
for epoch in range(num_epochs):
    print(str(epoch)+'epoch running')
    for batch_idx, (data, targets) in enumerate(tqdm(train_data_loader)):

        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_data_loader, model) * 100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_data_loader, model) * 100:.2f}")
