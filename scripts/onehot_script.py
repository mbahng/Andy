from src.data import *
from src.transformations import *
from src.models import *
from src.utils import *
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TAXONOMY_NAME = "subfamily"
ORDER_NAME = "Diptera"
FAMILY_NAME = "Phoridae"

t = GeneticOneHot(length=720, zero_encode_unknown=True, include_height_channel=False)
d = GeneticDataset("data/train.tsv", transform=t, drop_level=TAXONOMY_NAME, allowed_classes=[("order", [ORDER_NAME]), ("family", ["Cecidomyiidae"])])
d_val = GeneticDataset("data/test.tsv", transform=t, drop_level=TAXONOMY_NAME, allowed_classes=[("order", [ORDER_NAME]), (TAXONOMY_NAME, d.get_classes(TAXONOMY_NAME)[0])])

train_set = d
val_set = d_val

# train_set, val_set = random_split(d, [.8, .2])

train_dl = DataLoader(train_set, batch_size=128, shuffle=True)
test_dl = DataLoader(val_set, batch_size=32, shuffle=True)

classes, sizes = d.get_classes(TAXONOMY_NAME)

model = GeneticCNN1D(720, len(classes))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Balanced loss function
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
        for data in test_dl:
            inputs, labels = data
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
    print(f"Accuracy Scores: {accuracy}")

# Save the model
torch.save(model.state_dict(), "models/onehot_diptera_subfamily.pth")
