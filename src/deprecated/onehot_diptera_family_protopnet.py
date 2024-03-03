"""This scripts trains a black box model on Genetic Data, saves it. Then trains a ProtoPNet model."""

from preprocessing.datasets import *
from transformations import GeneticOneHot
from models import *
from utils import *
from torch.utils.data import DataLoader, random_split
from utils.log import create_logger


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Configuration
TAXONOMY_NAME = "family"
ORDER_NAME = "Diptera"
CHOP_LENGTH = 720
# The blackbox model will be saved to model_dir/{FILE_NAME_BASE}_{EPOCH}.pth
FILE_NAME_BASE = "onehot_128_4conv_30wide_40prototypes_per_class"

model_dir = "./saved_models/cnn"

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

conv1 = nn.Conv2d(4, 16, kernel_size=(1,3), padding=(0,1))


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

    # Save the model
    if epoch >= 5:
        torch.save(model.state_dict(), f"pretrained_models/{FILE_NAME_BASE}_{epoch + 1}.pth")

log("Finished Blackbox Training")

# =============================================================================

# Now, the protopnet implementation

from torch.utils.data import DataLoader
import torch
import src.train_and_test as tnt
import src.utils.save as save

import os

TAXONOMY_NAME = "family"
 
prototypes_per_class = 40
# latent_size = (128, 1, 720 // 8)
latent_size = 720 // 8


# Usually, you don't want to retrain the blackbox model, so you can just load it from the last epoch. This is the file name base for the blackbox model that you want to use.
# Just uncomment this line and the break statement in the blackbox training loop to skip that step.
# OLD_FILE_NAME_BASE = "onehot_128_4conv_30wide_20prototypes_per_class"

OLD_FILE_NAME_BASE = FILE_NAME_BASE

ppnet = construct_genetic_ppnet(720, len(classes), (prototypes_per_class*len(classes),128, 1,4), f"models/Blackbox/{OLD_FILE_NAME_BASE}_{7}.pth")

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from src.utils.settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from src.utils.settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from src.utils.settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from src.utils.settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from src.utils.settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

print("Start Training")
for epoch in range(num_train_epochs):
    print('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_dl, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_dl, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    
    accu = tnt.test(model=ppnet_multi, dataloader=val_dl,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                            target_accu=0.90, log=log)

    print(f"Accuracy: {accu}")
