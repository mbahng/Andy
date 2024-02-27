"""This script trains a model to classify families of Diptera using a one-hot encoding of the genetic data."""

from src.data import *
from src.transformations import GeneticOneHot
from src.models import *
from src.utils import *
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TAXONOMY_NAME = "family"
ORDER_NAME = "Diptera"
CHOP_LENGTH = 720
FILE_NAME_BASE = "onehot_blackbox_128_3conv"

# Create an instance of the transformation class
t = GeneticOneHot(length=CHOP_LENGTH, zero_encode_unknown=True, include_height_channel=True)

# NOTE: Data files are available in the Google Drive. They're split 80/20 into train and test sets, from samples in the restrictive BIOSCAN dataset.

# Create a training dataset, dropping all samples that don't have a family label, and only allowing the order Diptera
d_train = GeneticDataset("data/BIOSCAN-1M/small_diptera_family-train.tsv", transform=t, drop_level=TAXONOMY_NAME, allowed_classes=[("order", [ORDER_NAME])])

# Create a test dataset, dropping all samples that don't have a family label, only allowing the order Diptera, and only allowing the families present in the training set
d_val = GeneticDataset("data/BIOSCAN-1M/small_diptera_family-validation.tsv", transform=t, drop_level=TAXONOMY_NAME, allowed_classes=[("order", [ORDER_NAME]), ("family", d_train.get_classes(TAXONOMY_NAME)[0])])

# Create data loaders
train_dl = DataLoader(d_train, batch_size=128, shuffle=True)
val_dl = DataLoader(d_val, batch_size=32, shuffle=True)

classes, sizes = d_train.get_classes(TAXONOMY_NAME)
print(f"Classes: {classes}")

# Create the model
model = GeneticCNN2D(CHOP_LENGTH, len(classes))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Weight the loss function to account for class imbalance
class_weights = 1 / torch.tensor(sizes, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

# Training loop
EPOCHS = 10

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct_guesses = 0
    total_guesses = 0
    model.train()

    for i, data in enumerate(train_dl, 0):
        inputs, labels = data
        # Convert labels to integers
        labels = [classes.index(l) for l in labels[taxonomy_level_index_map[TAXONOMY_NAME]]]
        labels = torch.tensor(labels, dtype=torch.long)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        y_pred = torch.argmax(outputs, dim=1)
        correct_guesses += torch.sum(y_pred == labels)
        total_guesses += len(y_pred)

        if i % 10 == 0:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f} accuracy: {correct_guesses / total_guesses}")
            running_loss = 0.0
    
    # Evaluate on test set with balanced accuracy
    model.eval()
    correct_guesses = [0 for _ in range(len(classes))]
    total_guesses = [0 for _ in range(len(classes))]

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data
            # Convert labels to the same integers as the training set
            labels = [classes.index(l) for l in labels[taxonomy_level_index_map[TAXONOMY_NAME]]]
            labels = torch.tensor(labels, dtype=torch.long)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            y_pred = torch.argmax(outputs, dim=1)

            for i in range(len(classes)):
                correct_guesses[i] += torch.sum((y_pred == labels) & (labels == i))
                total_guesses[i] += torch.sum(labels == i)
    
    accuracy = [correct_guesses[i] / max(1, total_guesses[i]) for i in range(len(classes))]
    balanced_accuracy = sum(accuracy) / len(classes)
    print(f"Epoch {epoch + 1} balanced accuracy: {balanced_accuracy}")
    print(f"Accuracy by class: {accuracy}")

    # Save the model
    if epoch > 5:
        torch.save(model.state_dict(), f"models/{FILE_NAME_BASE}_{epoch + 1}.pth")