import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

testdataload = dataset
#train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader_tensor = DataLoader(testdataload, batch_size=10, shuffle=False)


fullmodel.eval()  # Set to evaluation mode

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader_tensor:  # test_loader contains the test dataset
        outputs = fullmodel(images)
        _, preds = torch.max(outputs, 1)
        all_predictions.extend(preds.numpy())
        all_labels.extend(labels.numpy())

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_predictions))