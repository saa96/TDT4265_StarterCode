import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
import numpy as np

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    
    mean = np.mean(X_train, axis=(0,1))
    std = np.std(X_train, axis=(0,1))

    X_train = pre_process_images(X_train, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Use improved weight init
    print("Improved weights")
    use_improved_weight_init = True
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_init, val_history_improved_init = trainer_shuffle.train(
        num_epochs)
    
    # Use improved sigmoid
    print("Weights and sigmoid")
    use_improved_sigmoid = True
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_sigmoid, val_history_improved_sigmoid = trainer_shuffle.train(
        num_epochs)
    
    # Use momentum
    print("Weights, sigmoid and momentum")
    use_momentum = False
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_momentum, val_history_momentum = trainer_shuffle.train(
        num_epochs)
    
    plt.subplot(1, 2, 1)
    utils.plot_loss(val_history["loss"],
                    "Task 2 Model - No improvements", npoints_to_average=10)
    utils.plot_loss(
        val_history_improved_init["loss"], "Task 2 Model - Only improved weight", npoints_to_average=10)
    utils.plot_loss(
        val_history_improved_sigmoid["loss"], "Task 2 Model - Weight and sigmoid", npoints_to_average=10)
    utils.plot_loss(
        val_history_momentum["loss"], "Task 2 Model - Weights, sigmoid and momentum", npoints_to_average=10)
    utils.plot_loss(train_history_momentum["loss"],
        "Training Loss with improvements", npoints_to_average=10)
    plt.ylabel("Cross Entropy Loss - Average")
    plt.ylim([0, 1.25])
    plt.subplot(1, 2, 2)
    plt.ylim([0.70, 1])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model - No improvements")
    utils.plot_loss(
        val_history_improved_sigmoid["accuracy"], "Task 2 Model - Only improved weight")
    utils.plot_loss(
        val_history_improved_init["accuracy"], "Task 2 Model - Weight and sigmoid")
    utils.plot_loss(
        val_history_momentum["accuracy"], "Task 2 Model - Weight, sigmoid and momentum")
    utils.plot_loss(train_history_momentum["accuracy"], "Training Accuracy with improvements")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task3_train_loss.png")
    plt.show()
    