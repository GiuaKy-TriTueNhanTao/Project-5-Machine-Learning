from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import backend
import models
import train


OUTPUT_DIR = Path(__file__).resolve().parent


def save_confusion_matrix(actual, predicted, labels, title, filename):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of examples")
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return matrix


def generate_question_1():
    model = models.PerceptronModel(3)
    dataset = backend.PerceptronDataset(model)
    train.train_perceptron(model, dataset)

    weights = model.get_weights().detach().flatten().numpy()
    predictions = model(torch.tensor(dataset.x, dtype=torch.float32)).detach().numpy()
    accuracy = np.mean(np.where(predictions >= 0, 1, -1) == dataset.y.flatten())

    fig, ax = plt.subplots(figsize=(8, 6))
    positive = dataset.y.flatten() == 1
    ax.scatter(dataset.x[positive, 0], dataset.x[positive, 1], color="red", marker="+", label="+1")
    ax.scatter(dataset.x[~positive, 0], dataset.x[~positive, 1], color="blue", marker="_", label="-1")
    limits = np.array([-3.0, 3.0])
    if weights[1] != 0:
        ax.plot(limits, (-weights[0] * limits - weights[2]) / weights[1], color="black")
    elif weights[0] != 0:
        ax.axvline(-weights[2] / weights[0], color="black")

    ax.set(xlim=limits, ylim=limits, xlabel="x1", ylabel="x2")
    ax.set_title(f"Q1 Perceptron - Training Accuracy: {accuracy:.2%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "question-1-result.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Q1: training accuracy {accuracy:.2%}")


def generate_question_2():
    model = models.RegressionModel()
    dataset = backend.RegressionDataset(model)
    train.train_regression(model, dataset)

    inputs = torch.tensor(dataset.x, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(inputs).numpy().flatten()
    actual = dataset.y.flatten()
    mse = np.mean((predictions - actual) ** 2)
    order = np.argsort(dataset.x.flatten())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dataset.x.flatten()[order], actual[order], color="blue", label="sin(x)")
    ax.plot(dataset.x.flatten()[order], predictions[order], color="red", label="Model prediction")
    ax.set(xlabel="x", ylabel="y", title=f"Q2 Regression - MSE: {mse:.6f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "question-2-result.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Q2: mean squared error {mse:.6f}")


def generate_question_3():
    model = models.DigitClassificationModel()
    dataset = backend.DigitClassificationDataset(model)
    train.train_digitclassifier(model, dataset)

    with torch.no_grad():
        logits = model(torch.as_tensor(dataset.test_images, dtype=torch.float32))
    predicted = logits.argmax(dim=1).cpu().numpy()
    actual = np.asarray(dataset.test_labels)
    accuracy = np.mean(predicted == actual)
    save_confusion_matrix(
        actual,
        predicted,
        list(range(10)),
        f"Q3 Digit Classification - Test Accuracy: {accuracy:.2%}",
        "question-3-result.png",
    )
    print(f"Q3: test accuracy {accuracy:.2%}")


def generate_question_4():
    model = models.LanguageIDModel()
    dataset = backend.LanguageIDDataset(model)
    train.train_languageid(model, dataset)

    _, predicted, actual = dataset._predict(split="test")
    predicted = np.asarray([int(item) for item in predicted])
    actual = np.asarray(actual, dtype=int)
    accuracy = np.mean(predicted == actual)
    save_confusion_matrix(
        actual,
        predicted,
        [str(name) for name in dataset.language_names],
        f"Q4 Language Identification - Test Accuracy: {accuracy:.2%}",
        "question-4-result.png",
    )
    print(f"Q4: test accuracy {accuracy:.2%}")


def generate_question_5():
    model = models.DigitConvolutionalModel()
    dataset = backend.DigitClassificationDataset2(model)
    train.Train_DigitConvolution(model, dataset)

    with torch.no_grad():
        logits = model(torch.as_tensor(dataset.test_images, dtype=torch.float32))
    predicted = logits.argmax(dim=1).cpu().numpy()
    actual = np.asarray(dataset.test_labels)
    accuracy = np.mean(predicted == actual)
    save_confusion_matrix(
        actual,
        predicted,
        list(range(10)),
        f"Q5 Convolution - Test Accuracy: {accuracy:.2%}",
        "question-5-result.png",
    )
    print(f"Q5: test accuracy {accuracy:.2%}")


def main():
    backend.use_graphics = False
    np.random.seed(0)
    torch.manual_seed(0)

    for generate in (
        generate_question_1,
        generate_question_2,
        generate_question_3,
        generate_question_4,
        generate_question_5,
    ):
        generate()


if __name__ == "__main__":
    main()
