from torchvision import models as models
import torch.nn as nn
def model(pretrained, requires_grad):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # Modify the first convolutional layer to accept 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 360 classes in total
    model.fc = nn.Linear(2048, 360)
    return model