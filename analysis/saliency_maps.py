import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def compute_saliency_map(model, image, class_index=None):
    """Vanilla-gradient saliency map: the gradient of the predicted class
    score with respect to the input pixels (Simonyan et al., 2013), showing
    which pixels most influenced the model's decision."""
    image = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[0, class_index]
    gradients = tape.gradient(class_score, image)[0]
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    return saliency.numpy()


def plot_saliency_comparison(models, model_names, image, true_label, save_path=None):
    """Plot an input image next to each model's saliency map, for comparing
    how differently each architecture attends to the same input. Saliency
    concentrated on face/artifact regions suggests the model learned
    deepfake-relevant features; saliency scattered across the background
    is evidence of overfitting to dataset-specific noise."""
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4 * (len(models) + 1), 4))

    axes[0].imshow(image)
    axes[0].set_title(f"Input (label: {true_label})")
    axes[0].axis("off")

    for ax, model, name in zip(axes[1:], models, model_names):
        saliency = compute_saliency_map(model, image)
        ax.imshow(saliency, cmap="hot")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def generalization_gap(model, X_primary, y_primary, X_secondary, y_secondary):
    """Compare validation accuracy and cross-entropy loss on the primary
    dataset vs. a second, held-out dataset, to quantify how much of a
    model's reported accuracy is dataset-specific rather than general."""
    primary_loss, primary_acc = model.evaluate(X_primary, y_primary, verbose=0)
    secondary_loss, secondary_acc = model.evaluate(X_secondary, y_secondary, verbose=0)

    print(f"Primary dataset   - accuracy: {primary_acc:.4f}, cross-entropy loss: {primary_loss:.4f}")
    print(f"Secondary dataset - accuracy: {secondary_acc:.4f}, cross-entropy loss: {secondary_loss:.4f}")
    print(f"Generalization gap (accuracy drop): {primary_acc - secondary_acc:.4f}")

    return {
        "primary_accuracy": primary_acc,
        "primary_loss": primary_loss,
        "secondary_accuracy": secondary_acc,
        "secondary_loss": secondary_loss,
        "generalization_gap": primary_acc - secondary_acc,
    }


if __name__ == "__main__":
    # Example usage, once trained models and test data are loaded:
    #
    # models = [custom_model, resnet_model, vgg16_model, densenet_model]
    # model_names = ["Custom CNN", "ResNet50", "VGG16", "DenseNet121"]
    # plot_saliency_comparison(models, model_names, X_test_cnn[0], y_test_cnn[0])
    # generalization_gap(custom_model, X_test_cnn, y_test_cnn, X_secondary, y_secondary)
    pass
