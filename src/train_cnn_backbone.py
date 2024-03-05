from preprocessing.datasets import *
from preprocessing.transformations import GeneticOneHot
from model.model import *
from utils import *
from torch.utils.data import DataLoader
from utils.log import create_logger
from utils.helpers import makedir

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Configuration
TAXONOMY_NAME = "family"
ORDER_NAME = "Diptera"
CHOP_LENGTH = 720
# The blackbox model will be saved to model_dir/{FILE_NAME_BASE}_{EPOCH}.pth
FILE_NAME_BASE = "4conv-128"

model_dir = os.path.join("saved_models", "bioscan_cnn_backbone")
makedir(model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, f'{FILE_NAME_BASE}.log'))

# Create an instance of the transformation class
t = GeneticOneHot(length=CHOP_LENGTH, zero_encode_unknown=True, include_height_channel=True)

 # Create a training dataset from the small_diptera_family-train.tsv file. drop_level and allowed_classes don't matter here since the dataset is already well behaved.
d_train = GeneticDataset(os.path.abspath("data/BIOSCAN-1M/small_diptera_family-train.tsv"), transform=t, drop_level=TAXONOMY_NAME, allowed_classes=[("order", [ORDER_NAME])], one_label=TAXONOMY_NAME)
classes, sizes = d_train.get_classes(TAXONOMY_NAME)

# Create a test dataset, dropping all samples that don't have a family label, only allowing the order Diptera, and only allowing the families present in the training set
d_val = GeneticDataset(os.path.abspath("data/BIOSCAN-1M/small_diptera_family-validation.tsv"), transform=t, drop_level=TAXONOMY_NAME, allowed_classes=[("order", [ORDER_NAME]), ("family", d_train.get_classes(TAXONOMY_NAME)[0])], one_label=TAXONOMY_NAME, classes=classes)

# Create data loaders
train_dl = DataLoader(d_train, batch_size=128, shuffle=True)
val_dl = DataLoader(d_val, batch_size=32, shuffle=True)

# Create the model
model = GeneticCNN2D(CHOP_LENGTH, len(classes), include_connected_layer=True).cuda() 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Weight the loss function to account for class imbalance
class_weights = 1 / torch.tensor(sizes, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

# Training loop
EPOCHS = 7

for epoch in range(EPOCHS):
    # break
    running_loss = 0.0
    correct_guesses = 0
    total_guesses = 0
    model.train()

    for i, data in enumerate(train_dl, 0):
        inputs, labels = data
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
            log(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f} accuracy: {correct_guesses / total_guesses}")
            running_loss = 0.0
    
    # Evaluate on test set with balanced accuracy
    model.eval()
    correct_guesses = [0 for _ in range(len(classes))]
    total_guesses = [0 for _ in range(len(classes))]

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            y_pred = torch.argmax(outputs, dim=1)

            for i in range(len(classes)):
                correct_guesses[i] += torch.sum((y_pred == labels) & (labels == i))
                total_guesses[i] += torch.sum(labels == i)
    
    accuracy = [correct_guesses[i] / max(1, total_guesses[i]) for i in range(len(classes))]
    balanced_accuracy = sum(accuracy) / len(classes)
    log(f"Epoch {epoch + 1} balanced accuracy: {balanced_accuracy}")


torch.save(model.state_dict(), os.path.join("saved_models", "bioscan_cnn_backbone", f"{FILE_NAME_BASE}_ep{EPOCHS}.pth"))

log("Finished Blackbox Training")


