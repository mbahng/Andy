import torch
from src.data import * 
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel 
from open_clip import create_model_from_pretrained, create_model_and_transforms

dataset = CubDataset()

device = ("cuda" if torch.cuda.is_available() else "cpu")

def extract(train_images, text_list, verbose):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)    # type: ignore
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    output_list = []
    b_size = 50
    num_iter = (
        len(train_images) + b_size - 1
    ) // b_size  # Use ceiling division to handle the last batch

    # Go through each batch and extract features
    for i in range(num_iter):
        if verbose: 
            print(f"  Extract Iteration {i+1}/{num_iter+1}")
        if i == num_iter - 1:
            # batch_range = slice(i * 100, len(train_images))
            batch_range = slice(i * b_size, len(train_images))
        else:
            batch_range = slice(i * b_size, (i + 1) * b_size)
            # batch_range = slice(i * 100, (i + 1) * 100)
        # Prepare inputs for the CLIP model
        inputs = processor(
            text=text_list,
            images=train_images[batch_range],
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Get model outputs without gradients
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the image-text similarity scores and convert to probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        output_list.append(probs.cpu())  # Move to CPU to avoid out of memory on GPU

    # Concatenate all batch outputs
    output = torch.cat(output_list)

    # Clean up to free GPU memory
    del model, processor, inputs, outputs, logits_per_image, probs      # type: ignore

    return output

text_labels = [f"a photo of a {cls}" for cls in dataset.classes]
b_size = 5000
num_iter = (len(dataset) + b_size - 1) // b_size

output_list = []
for i in range(num_iter):
    print(f"Data Iteration {i+1}/{num_iter+1})")
    print("==============================")
    if i == num_iter - 1:
        low, high = i * b_size, len(dataset)
    else:
        low, high = i * b_size, (i + 1) * b_size

    train_images = [dataset[i][0].convert("RGB") for i in range(low, high)]

    # Extract features using the CLIP-based model
    output = extract(train_images, text_labels, True)
    output_list.append(output)

    del train_images, output 
   

output_list = torch.cat(output_list)
print(f"Shape of Output List: {output_list.shape}")
torch.save(output_list, os.path.join("output", "clip_cub_train.pt"))

predictions = torch.Tensor(output_list.argmax(dim=1))
actual = torch.tensor([dataset[i][1] for i in range(len(dataset))])

accuracy = (predictions == actual).sum().item() / len(actual) 

print(f"Accuracy : {accuracy}")


