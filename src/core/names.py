from typing import Literal


dataset_names = Literal[
    "mnist", "cifar10", "cifar100", "fashion_mnist", "imagenet", "mini_imagenet"
]
model_names = Literal[
    "example_cnn",
    "le_net",
    "alex_net",
    "vgg_net11",
    "vgg_net13",
    "vgg_net16",
    "vgg_net19",
    "res_net18",
    "res_net34",
    "res_net50",
    "res_net101",
    "res_net152",
    "inception_v1",
    "inception_v2",
    "dense_net_cifar",
    "dense_net121",
    "dense_net169",
    "dense_net201",
    "dense_net264",
    "mobile_net",
    "shuffle_net_v1",
]
loss_function_names = Literal["cross_entropy", "mse", "nll_loss"]
optimizer_names = Literal["sgd", "adam", "adamw"]
