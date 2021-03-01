import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
dataloaders import load_cifar10
trainer import Trainer, compute_loss_and_accuracy


class Task3Model(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        # TODO: Implement this function (Task  2a)
        num_filters = 8  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # [in: 3, out 8, input spatial size: 32x32, output spatial size: 32 x 32 ]
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
                ),
            nn.ELU(),
            # [in: 8, out 16, input spatial size: 32x32, output spatial size: 16 x 16]
            nn.Conv2d(
                in_channels=num_filters,
                out_channels= num_filters * 2,
                kernel_size=5,
                stride=1,
                padding=2
                ),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
                ),
            # [in: 16, out 16 , input spatial size: 16x16, output spatial size: 16x16]
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*2,
                kernel_size=5,
                stride=1,
                padding=2
                ),
            nn.ELU(),
            # [in: 16, out 32 , input spatial size: 16x16, output spatial size: 8x8]
            nn.Conv2d(
               in_channels=num_filters*2,
               out_channels=num_filters*4,
               kernel_size=5,
               stride=1,
               padding=2
               ),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
                ),
            #  [in: 32, out 32 , input spatial size: 8x8, output spatial size: 8x8]
            nn.Conv2d(
                in_channels=num_filters * 4,
                out_channels=num_filters * 4,
                kernel_size=5,
                stride=1,
                padding=2
                ),
            nn.ELU(),
            #  [in: 32, out 32, input spatial size: 8x8, output spatial size: 4x4]
            nn.Conv2d(
                in_channels=num_filters * 4,
                out_channels=num_filters * 4,
                kernel_size=5,
                stride=1,
                padding=2
                ),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
                ),
            nn.Flatten()

        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        #self.num_output_features = 32*32*32
        self.num_output_features = 32 * 4 * 4
        #self.num_output_features = 128 * 4 * 4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.num_output_features, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=num_classes),

        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        batch_size = x.shape[0]
        out = x
        #print(out.shape)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Task3Model(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task3testing")

    #Loading best model and plotting train, val, test accuracy
    trainer.load_best_model()
    print(trainer.model)
    train_loss, train_acc = compute_loss_and_accuracy(
        dataloaders[0], trainer.model, loss_criterion=nn.CrossEntropyLoss()
    )
    val_loss, val_acc = compute_loss_and_accuracy(
        dataloaders[1], trainer.model, loss_criterion=nn.CrossEntropyLoss()
    )

    test_loss, test_accuracy = compute_loss_and_accuracy(
        dataloaders[2], trainer.model, loss_criterion=nn.CrossEntropyLoss()
    )
    print('Performance for best model:')
    print('train loss, ', train_loss)
    print('train accuracy: ', train_acc)
    print('validation loss: ', val_loss)
    print('validation accuracy: ', val_acc)
    print('test_loss: ', test_loss)
    print('test_accuracy: ', test_accuracy)
