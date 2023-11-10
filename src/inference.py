import models
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)



# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the model
model = models.model(pretrained=False, requires_grad=False).to(device)
# load the model checkpoint
checkpoint = torch.load('../outputs/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

train_csv = pd.read_csv('../input/drug-classifier/drug_CNN/train_drugidx.csv')
genres = train_csv.columns.values[2:]
# prepare the test dataset and dataloader
test_data = ImageDataset(
    train_csv, train=False, test=True
)
test_loader = DataLoader(
    test_data, 
    batch_size=1,
    shuffle=False
)

# Initialize arrays to store true and predicted labels
true_labels = []
predicted_labels = []
# Initialize a list to store losses
losses = []

# Define the loss function
loss_function = models.focal_binary_cross_entropy #nn.BCELoss()

for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label'].to(device)
    # Forward pass
    outputs = model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu().numpy()
    if isinstance(outputs, np.ndarray):
        outputs_loss = torch.tensor(outputs).to(device).float()
    else:
        outputs_loss = outputs.float().to(device)
    
    if isinstance(target, np.ndarray):
        target = torch.tensor(target).to(device).float()
    else:
        target = target.float().to(device)
    
    # Compute the loss
    loss = loss_function(outputs_loss, target.float())  # Make sure target is a float tensor
    losses.append(loss.item())
    
    # Convert outputs to binary predictions
    predicted = (outputs > 0.00001).astype(int)
    
    # Store predictions and true labels
    true_labels.append(target)
    predicted_labels.append(predicted)
    
    # Convert the predictions and targets to the correct format for metric calculation
    target_indices = np.where(target[0].cpu() == 1)[0]
    predicted_indices = np.where(predicted[0] == 1)[0]
    
    string_predicted = ' '.join(genres[predicted_indices])
    string_actual = ' '.join(genres[target_indices])
    
    image = image.squeeze(0).detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    print(outputs)
    print(genres[np.argmax(outputs)], genres[target_indices])
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    plt.savefig(f"../outputs/inference_{counter}.jpg")
    plt.show()

# Concatenate all the arrays
# Convert all elements to NumPy arrays, assuming they are tensors on the device
true_labels = np.vstack([label.cpu().numpy() if isinstance(label, torch.Tensor) else label for label in true_labels])
predicted_labels = np.vstack([label.cpu().numpy() if isinstance(label, torch.Tensor) else label for label in predicted_labels])


# Calculate metrics
precision_micro = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall_micro = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
f1_micro = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

print(f"Precision Micro: {precision_micro:.4f}")
print(f"Precision Macro: {precision_macro:.4f}")
print(f"Recall Micro: {recall_micro:.4f}")
print(f"Recall Macro: {recall_macro:.4f}")
print(f"F1 Micro: {f1_micro:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")

# After the loop, you can calculate the average loss
average_loss = sum(losses) / len(losses)
print(f"Average Loss: {average_loss:.4f}")