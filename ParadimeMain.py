from ParadimeNAM import twoNAMHybrid
import numpy as np
import torch as th
import torchvision
import paradime
from matplotlib import pyplot as plt
from sklearn import datasets
import json

iris = datasets.load_iris()
mnist = torchvision.datasets.MNIST(
    '../data',
    train=False,
    download=False,
)
mnist_data = mnist.data.reshape(-1, 28*28) / 255.
num_items = 5000
mnist_subset = mnist_data[:num_items]
target_subset = mnist.targets[:num_items]
weights = [0.0, 5.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0]
test_accuracies = np.load('test_accuracies.npy')
train_accuracies = np.load('train_accuracies.npy')
loss_history = th.load("losses.pt")
losses = {"tsne": loss_history, "class": paradime.loss.ClassificationLoss()}


tsne = paradime.routines.ParametricTSNE(in_dim=28*28)
global_rel = tsne.global_relations
batch_rel = tsne.batch_relations


# Loading and evaluating phase:
for w in weights:
    print("------------")
    model = twoNAMHybrid(input_dim=28 * 28, hidden_dim=100, num_classes=10, output_dim=2)
    model.load_state_dict(th.load(f'hybrid_tsne_model_{w}.pth'))
    model.eval()

    # Create a new ParametricDR with the loaded model
    hybrid_tsne = paradime.dr.ParametricDR(
        model=model,
        global_relations=global_rel,
        batch_relations=batch_rel,
        losses=losses,
        use_cuda=True,
        verbose=True,
    )
    hybrid_tsne.train({
        "main": mnist_subset,
        "labels": target_subset,
    })


    # Generate embeddings
    embeddings = hybrid_tsne.apply(mnist_subset, "embed")

    # Make classifications
    train_logits = hybrid_tsne.apply(mnist_subset, "classify")
    train_prediction = th.argmax(train_logits, dim=1)

    # You might want to convert the PyTorch tensors to NumPy arrays for plotting
    embeddings_np = embeddings.detach().numpy()
    train_prediction_np = train_prediction.detach().numpy()

fig = plt.figure(figsize=(15, 15))

for i, (emb, w) in enumerate(zip(embeddings, weights)):
    ax = fig.add_subplot(3, 3, i + 1)

    paradime.utils.plotting.scatterplot(
        emb,
        labels=target_subset,
        ax=ax,
        legend=(i == 0),
        legend_options={"loc": 3},
    )
    ax.set_title(f"w_emb / w_class = {w}")

palette = paradime.utils.plotting.get_color_palette()
ax = fig.add_subplot(3, 3, 9)
ax.plot(weights, train_accuracies, c=palette["petrol"])
ax.plot(weights, test_accuracies, c=palette["aqua"])
ax.set_xscale("log")
ax.set_ylim([0,1])
ax.legend(["train", "test"])
ax.set_title("Classification accuracy");