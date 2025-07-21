from typing import Literal


dataset_names = Literal["mnist", "cifar10", "cifar100", "fashion_mnist", "imagenet"]
model_names = Literal[
    "example_cnn", "le_net", "alex_net", "vgg11", "vgg13", "vgg16", "vgg19"
]
loss_function_names = Literal["cross_entropy", "mse", "nll_loss"]
optimizer_names = Literal["sgd", "adam", "adamw"]
