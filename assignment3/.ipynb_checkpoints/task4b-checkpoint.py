
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
import pathlib
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image




def show_activation_and_filter_from_layer(layer_activation, indices,name:str, layer = None, plotfilters = False):
    """

    Args:
        indices: Indexes of which filters' activations to be visualized
        layer: The layer which has the filters we want to show. Cannot be None if plotfilters is true
        plotfilters: A boolean that determines if we also want to show the filters

    """
    activation_images = []
    filter_images = []
    print(model.conv1.weight.data[14].shape)
    for index in indices:
        activation_images.append(torch_image_to_numpy(layer_activation[0][index]))
        if plotfilters:
            try:
                filter_images.append(torch_image_to_numpy(layer.weight.data[index]))
            except AttributeError:
                print('plotfilters is True, but no layer was passed as argument')
                break

    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(20, 8)) if plotfilters else plt.figure(figsize=(20, 4))
    for i in range(len(activation_images)):
        if plotfilters:
            fig.add_subplot(2, len(indices), i + 1)
            plt.title(f"Filter at index {i}")
            plt.imshow(filter_images[i])
            fig.add_subplot(2,len(indices),len(indices) + i+1)
            plt.title(f"Feature map at index {i}")
            plt.imshow(activation_images[i], cmap='gray')
        else:
            fig.add_subplot(1, len(indices), i + 1)
            plt.title(f"Feature map at index {i}")
            plt.imshow(activation_images[i], cmap='gray')
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


first_layer_filter_indices = [14, 26, 32, 49, 52]

show_activation_and_filter_from_layer(activation, first_layer_filter_indices, 'task4b', first_conv_layer, plotfilters=True)

'''Task 4b '''


def get_activation_from_last_layer():
    last_activation = image
    for layer in model.children():
        if layer == model.layer4:
            last_activation = layer(last_activation)
            return last_activation
        else:
            last_activation = layer(last_activation)


last_layer_filter_indices = [i for i in range(10)]
last_layer_activation = get_activation_from_last_layer()
show_activation_and_filter_from_layer(last_layer_activation,last_layer_filter_indices, 'task4c')
