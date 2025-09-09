import logging

import matplotlib.pyplot as plt

from core.config import DiffusionConfig
from core.device import available_device
from core.logger import init_logger
from core.optimizer import AdamWParams
from core.pipeline import DiffusionPipeline
from core.model_params.diffusion_model_params import (
    DDPMParams,
)


logger = logging.getLogger("main")


def main():
    init_logger("INFO")

    logger.info("Starting Deep Learning Models")
    logger.info(f"Available device: {available_device()}")

    # config = ClassificationConfig(
    #     name="example_cnn",
    #     model="example_cnn",
    #     dataset="mnist",
    #     batch_size=64,
    #     optimizer="adam",
    #     optimizer_params=AdamParams(lr=0.001),
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("example_cnn")

    # config = ClassificationConfig(
    #     name="le_net",
    #     model="le_net",
    #     dataset="mnist",
    #     batch_size=64,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("le_net")

    # config = ClassificationConfig(
    #     name="alex_net",
    #     model="alex_net",
    #     dataset="mini_imagenet",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("alex_net")

    # config = ClassificationConfig(
    #     name="vgg11",
    #     model="vgg11",
    #     dataset="imagenet",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("vgg11")

    # config = ClassificationConfig(
    #     name="res_net18",
    #     model="res_net18",
    #     dataset="mini_imagenet",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=3,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("res_net18")

    # config = ClassificationConfig(
    #     name="res_net50",
    #     model="res_net50",
    #     dataset="mini_imagenet",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=3,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("res_net50")

    # config = ClassificationConfig(
    #     name="vgg_net_16_cifar10",
    #     model="vgg_net16",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("vgg_net_16_cifar10")

    # config = ClassificationConfig(
    #     name="res_net50_cifar10",
    #     model="res_net50",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=30,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("res_net50_cifar10")
    # 71.44%

    # config = ClassificationConfig(
    #     name="res_net32_padded_mnist2",
    #     model="res_net34",
    #     dataset="padded_mnist",
    #     batch_size=128,
    #     shuffle=True,
    #     epochs=10,
    #     optimizer="adamw",
    #     optimizer_params=AdamWParams(),
    #     loss_function="cross_entropy",
    # )
    # ClassificationConfig.save_config(config)

    # config = ClassificationConfig(
    #     name="inception_v1_cifar10",
    #     model="inception_v1",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=30,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("inception_v1_cifar10")
    # 70.78%

    # config = ClassificationConfig(
    #     name="inception_v2_cifar10",
    #     model="inception_v2",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=30,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("inception_v2_cifar10")
    # 74.07%

    # config = ClassificationConfig(
    #     name="dense_net_cifar10",
    #     model="dense_net_cifar",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     epochs=30,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("dense_net_cifar10")
    # 73.09%

    # config = ClassificationConfig(
    #     name="dense_net121_cifar10",
    #     model="dense_net121",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=30,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("dense_net121_cifar10")
    # 74.32%

    # config = ClassificationConfig(
    #     name="mobile_net_cifar10",
    #     model="mobile_net",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     epochs=30,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("mobile_net_cifar10")
    # 72.96%

    # config = ClassificationConfig(
    #     name="shuffle_net_v1_cifar10",
    #     model="shuffle_net_v1",
    #     dataset="cifar10",
    #     batch_size=64,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params={"lr": 0.001, "weight_decay": 1e-4},
    #     loss_function="cross_entropy",
    #     epochs=15,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("shuffle_net_v1_cifar10")

    # config = ClassificationConfig(
    #     name="efficient_net_v1_b0_cifar10",
    #     model="efficient_net_v1_b0",
    #     dataset="cifar10",
    #     batch_size=128,
    #     shuffle=True,
    #     optimizer="rmsprop",
    #     optimizer_params={"lr": 0.016, "weight_decay": 1e-5},
    #     loss_function="cross_entropy",
    #     early_stopping=True,
    #     early_stopping_min_delta=0.05,
    #     epochs=100,
    # )
    # ClassificationConfig.save_config(config)
    # config = ClassificationConfig.load_config("efficient_net_v0_b1_cifar10")
    # 77.92%

    # pipeline = ClassificationPipeline(config)
    # pipeline.run()

    # config = GANConfig(
    #     name="gan_mnist",
    #     model="gan",
    #     dataset="mnist",
    #     batch_size=128,
    #     shuffle=True,
    #     g_optimizer="adam",
    #     g_optimizer_params=AdamParams(lr=0.0002, betas=(0.5, 0.999)),
    #     d_optimizer="adam",
    #     d_optimizer_params=AdamParams(lr=0.000075, betas=(0.5, 0.999)),
    #     g_loss_function="bce_with_logits",
    #     d_loss_function="gan_discriminator_loss",
    #     epochs=30,
    #     real_label=0.95,
    # )
    # GANConfig.save_config(config)
    # # config = GANConfig.load_config("gan_mnist")

    # pipeline = GANPipeline(
    #     config,
    #     transforms.Compose(
    #         [
    #             transforms.Resize((28, 28)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5,), (0.5,)),
    #         ]
    #     ),
    # )
    # pipeline.run()

    # config = GANConfig(
    #     name="srgan_df2k_ost_small",
    #     model="srgan",
    #     dataset="df2k_ost_small",
    #     batch_size=32,
    #     shuffle=True,
    #     g_optimizer="adamw",
    #     g_optimizer_params=AdamWParams(
    #         lr=1e-4,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-2,
    #     ),
    #     d_optimizer="adamw",
    #     d_optimizer_params=AdamWParams(
    #         lr=5e-5,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-2,
    #     ),
    #     g_loss_function="srgan_generator_loss",
    #     d_loss_function="gan_discriminator_loss",
    #     epochs=2,
    #     save_after_n_epoch=True,
    #     save_after_n_epoch_period=1,
    #     real_label=0.95,
    #     early_stopping=True,
    #     early_stopping_monitor="train_loss",
    # )
    # GANConfig.save_config(config)
    # config = GANConfig.load_config("srgan_df2k_ost_small")

    # config = GANConfig(
    #     name="esrgan_df2k_ost_small",
    #     model="esrgan",
    #     model_params={
    #         "beta": 0.2,
    #         "rrdb_layers": 16,
    #     },
    #     dataset="df2k_ost_small",
    #     batch_size=32,
    #     shuffle=True,
    #     g_optimizer="adamw",
    #     g_optimizer_params=AdamWParams(
    #         lr=1e-4,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-2,
    #     ),
    #     d_optimizer="adamw",
    #     d_optimizer_params=AdamWParams(
    #         lr=5e-5,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-2,
    #     ),
    #     g_loss_function="srgan_generator_loss",
    #     d_loss_function="esrgan_discriminator_loss",
    #     epochs=2,
    #     epoch_save=True,
    #     epoch_save_period=1,
    #     real_label=0.95,
    #     early_stopping=True,
    #     early_stopping_monitor="train_loss",
    # )
    # GANConfig.save_config(config)
    # config = GANConfig.load_config("esrgan_df2k_ost_small")

    # config = GANConfig(
    #     name="esrgan_plus_df2k_ost_small",
    #     model="esrgan_plus",
    #     model_params={
    #         "beta": 0.2,
    #         "rrdrb_layers": 16,
    #     },
    #     dataset="df2k_ost_small",
    #     batch_size=32,
    #     shuffle=True,
    #     g_optimizer="adamw",
    #     g_optimizer_params=AdamWParams(
    #         lr=1e-4,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-2,
    #     ),
    #     d_optimizer="adamw",
    #     d_optimizer_params=AdamWParams(
    #         lr=5e-5,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-2,
    #     ),
    #     g_loss_function="srgan_generator_loss",
    #     d_loss_function="esrgan_discriminator_loss",
    #     epochs=2,
    #     save_after_n_epoch=True,
    #     save_after_n_epoch_period=1,
    #     real_label=0.95,
    #     early_stopping=True,
    #     early_stopping_monitor="train_loss",
    # )
    # GANConfig.save_config(config)
    # config = GANConfig.load_config("esrgan_plus_df2k_ost_small")

    # pipeline = GANPipeline(config)
    # pipeline.run()

    # config = GANConfig(
    #     name="real_esrgan_df2k_ost_extra_small",
    #     model="real_esrgan",
    #     model_params={
    #         "beta": 0.2,
    #         "rrdb_layers": 16,
    #         "image_size": 48,
    #     },
    #     dataset="df2k_ost_small",
    #     batch_size=64,
    #     shuffle=True,
    #     g_optimizer="adamw",
    #     g_optimizer_params=AdamWParams(
    #         lr=1e-4,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-2,
    #     ),
    #     d_optimizer="adamw",
    #     d_optimizer_params=AdamWParams(
    #         lr=5e-5,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-2,
    #     ),
    #     g_loss_function="srgan_generator_loss",
    #     d_loss_function="esrgan_discriminator_loss",
    #     epochs=50,
    #     save_after_n_epoch=True,
    #     save_after_n_epoch_period=1,
    #     real_label=0.95,
    #     early_stopping=True,
    #     early_stopping_monitor="train_loss",
    #     early_stopping_patience=5,
    #     early_stopping_min_delta=-0.002,
    #     early_stopping_min_delta_strategy="previous_proportional",
    #     pretrain=True,
    #     pretrain_epochs=5,
    #     save_pretrained=True,
    #     load_pretrained=True,
    # )
    # GANConfig.save_config(config)
    # config = GANConfig.load_config("real_esrgan_df2k_ost_small")

    # pipeline = GANPipeline(
    #     config,
    #     target_transform=transforms.Compose(
    #         [
    #             transforms.ToImage(),
    #             transforms.Resize((48, 48)),
    #             transforms.ToDtype(torch.float32, scale=True),
    #         ]
    #     ),
    #     input_transform=transforms.Compose(
    #         [
    #             transforms.ToImage(),
    #             transforms.Resize((48, 48)),
    #             transforms.ToDtype(torch.float32, scale=True),
    #         ]
    #     ),
    # )
    # pipeline.run()

    # config = DiffusionConfig(
    #     name="simple_vae_mnist",
    #     model="simple_vae",
    #     model_params=SimpleVAEParams(hidden_dim=200, latent_dim=20),
    #     dataset="mnist",
    #     batch_size=128,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params=AdamWParams(lr=3e-4),
    #     loss_function="vae_loss",
    #     epochs=30,
    # )
    # DiffusionConfig.save_config(config)

    # config = DiffusionConfig(
    #     name="cvae_mnist",
    #     model="cvae",
    #     model_params=CVAEParams(),
    #     dataset="mnist",
    #     batch_size=128,
    #     shuffle=True,
    #     optimizer="adam",
    #     optimizer_params=AdamWParams(lr=3e-4),
    #     loss_function="vae_loss",
    #     epochs=50,
    # )
    # DiffusionConfig.save_config(config)

    # config = DiffusionConfig(
    #     name="conditional_cvae_padded_mnist",
    #     model="conditional_cvae",
    #     model_params=ConditionalVAEParams(),
    #     dataset="padded_mnist",
    #     batch_size=128,
    #     shuffle=True,
    #     optimizer="adamw",
    #     optimizer_params=AdamWParams(lr=3e-4),
    #     loss_function="vae_loss",
    #     epochs=50,
    # )
    # DiffusionConfig.save_config(config)

    # config = DiffusionConfig(
    #     name="cfg_cvae_padded_mnist",
    #     model="cfg_cvae",
    #     model_params=CFGCVAEParams(),
    #     dataset="padded_mnist",
    #     batch_size=128,
    #     shuffle=True,
    #     optimizer="adamw",
    #     optimizer_params=AdamWParams(lr=3e-4),
    #     loss_function="vae_loss",
    #     epochs=50,
    # )
    # DiffusionConfig.save_config(config)

    # config = DiffusionConfig(
    #     name="cfg_vq_vae_padded_mnist",
    #     model="cfg_vq_vae",
    #     model_params=CFGVQVAEParams(),
    #     dataset="padded_mnist",
    #     batch_size=32,
    #     shuffle=True,
    #     optimizer="adamw",
    #     optimizer_params=AdamWParams(lr=3e-4),
    #     loss_function="vq_vae_loss",
    #     epochs=50,
    # )
    # DiffusionConfig.save_config(config)

    config = DiffusionConfig(
        name="ddpm_mnist",
        model="ddpm",
        model_params=DDPMParams(),
        dataset="mnist",
        validation=False,
        batch_size=128,
        shuffle=True,
        optimizer="adamw",
        optimizer_params=AdamWParams(
            lr=1e-4,
        ),
        loss_function="mse",
        epochs=100,
        save_after_n_epoch=True,
        save_after_n_epoch_period=10,
        early_stopping=False,
        early_stopping_monitor="val_loss",
        early_stopping_patience=5,
        early_stopping_min_delta=0.02,
        early_stopping_min_delta_strategy="delta_proportional",
    )
    DiffusionConfig.save_config(config)

    pipeline = DiffusionPipeline(config)
    # pipeline.train()

    # pipeline.load("last")
    # pipeline.evaluate(steps=1000, guidance_scale=2.5)

    # pipeline.evaluate(
    #     steps=1000, guidance_scale=1.0, file_postfix="8_cfg_1.0_seed_2025", seed=2025
    # )
    # pipeline.evaluate(
    #     steps=1000, guidance_scale=1.5, file_postfix="8_cfg_1.5_seed_2025", seed=2025
    # )
    # pipeline.evaluate(
    #     steps=1000, guidance_scale=2.0, file_postfix="8_cfg_2.0_seed_2025", seed=2025
    # )
    # pipeline.evaluate(
    #     steps=1000, guidance_scale=2.5, file_postfix="8_cfg_2.5_seed_2025", seed=2025
    # )
    # pipeline.evaluate(
    #     steps=1000, guidance_scale=3.0, file_postfix="8_cfg_3.0_seed_2025", seed=2025
    # )

    # for i in range(1, 11):
    #     pipeline.evaluate(
    #         steps=i * 100,
    #         guidance_scale=2.5,
    #         show=False,
    #         save_file_postfix=f"{i * 100}",
    #         seed=0,
    #     )

    # images = []
    # for i in range(1, 11):
    #     images.append(
    #         pipeline.model(
    #             steps=i * 100,
    #             guidance_scale=2.5,
    #             prompt=2,
    #             # file_name="3",
    #             # save=True,
    #             seed=0,
    #         )
    #     )
    # _, axes = plt.subplots(10, 1, figsize=(2, 20))
    # axes = axes.reshape(-1)
    # ax = axes[0]
    # ax.axis("off")
    # for i in range(10):
    #     img = images[i]
    #     print(img)
    #     ax.imshow(img, cmap="gray")
    #     ax.set_title((i + 1) * 100, fontsize=10, pad=2)
    # plt.tight_layout()
    # plt.savefig("ddpm_mnist_3")
    # _ = plt.show()
    # plt.close()

    for i in range(10, 101, 10):
        pipeline.load(f"epoch-{i}")

        pipeline.evaluate(
            steps=1000,
            guidance_scale=2.5,
            show=False,
            file_postfix=f"epoch-{i}-seed-25",
            seed=25,
            batch_size=16,
        )

        pipeline.clean()


if __name__ == "__main__":
    main()
