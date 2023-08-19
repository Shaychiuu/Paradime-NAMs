import pandas as pd
import numpy as np
import torch as th
import paradime
import paradime.dr
import paradime.loss
import paradime.routines
import paradime.utils
from matplotlib import pyplot as plt
from sklearn import datasets
import torchvision
import json


mnist = torchvision.datasets.MNIST(
    '../data',
    train=True,
    download=True,
)
mnist_data = mnist.data.reshape(-1, 28*28) / 255.
num_items = 5000

mnist_subset = mnist_data[:num_items]
target_subset = mnist.targets[:num_items]

class twoNAM(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, num_layers=1):
        super(twoNAM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.submodules = th.nn.ModuleList()

        # Create the submodules for each input feature
        for i in range(input_dim):
            submodule = th.nn.Sequential()
            # Add layers to the submodule
            for l in range(num_layers):
                if l == 0:
                    submodule.add_module(f"linear_{l}", th.nn.Linear(1, hidden_dim))
                else:
                    submodule.add_module(f"linear_{l}", th.nn.Linear(hidden_dim, hidden_dim))
                submodule.add_module(f"ELU_{l}", th.nn.ELU())
                submodule.add_module(f"dropout_{l}", th.nn.Dropout(0.5))

            # Add the output layer
            submodule.add_module(f"linear_{num_layers}", th.nn.Linear(hidden_dim, hidden_dim))
            self.submodules.append(submodule)

        # Add the final layer
        self.final_layer = th.nn.Linear(input_dim * hidden_dim, output_dim)

    def forward(self, x):
        # Initialize a list to store the outputs of submodules
        output = []
        for i in range(self.input_dim):
            # Compute the output of the i-th submodule and append it to the list
            output.append(self.submodules[i](x[:, i].unsqueeze(1)).squeeze())
        # Concatenate the outputs along the first dimension
        output = th.cat(output, dim=1)
        # Pass through the final layer to get 2D output
        output = self.final_layer(output)
        return th.sigmoid(output)

    def classify(self, x):
        return self.forward(x)

    def embed(self, x):
        return self.forward(x)

    def init_weights(self, m):
        if type(m) == th.nn.Linear:
            th.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_feature_maps(self, resolution=100):
        output = th.zeros(self.input_dim, resolution)

        for i in range(self.input_dim):
            for j in range(resolution):
                output[i, j] = self.submodules[i](th.tensor([[j / resolution]]))

        return output.detach().numpy()

class twoNAMHybrid(twoNAM):
    def __init__(self, input_dim, hidden_dim, num_classes, output_dim=2, num_layers=1):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers)
        self.class_layer = th.nn.Linear(output_dim, num_classes)
        self.alpha = th.nn.Parameter(th.tensor(1.0))

    def embed(self, x):
        return self.forward(x)

    def classify(self, x):
        x = self.embed(x)
        x = self.class_layer(x)
        return x


tsne = paradime.routines.ParametricTSNE(in_dim=28*28)
global_rel = tsne.global_relations
batch_rel = tsne.batch_relations
init_phase, main_phase = tsne.training_phases
rel_loss = main_phase.loss

embeddings = []
train_accuracies = []
test_accuracies = []

weights = [0.0, 5.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0]

losses = {"tsne": rel_loss, "class": paradime.loss.ClassificationLoss()}

for w in weights:
    paradime.utils.logging.log(f"Weight: {w}")
    print("xxxxx")

    hybrid_tsne = paradime.dr.ParametricDR(
        model=twoNAMHybrid(
            input_dim=28 * 28, hidden_dim=100, num_classes=10, output_dim=2,
        ),
        global_relations=global_rel,
        batch_relations=batch_rel,
        losses=losses,
        use_cuda=True,
        verbose=True,
    )
    hybrid_tsne.add_training_phase(
        epochs=50,
        batch_size=500,
        learning_rate=0.001,
        loss_keys=["tsne", "class"],
        loss_weights=[w, 1.0],
    )
    hybrid_tsne.train({
        "main": mnist_subset,
        "labels": target_subset,
    })

    # Save the model after training
    th.save(hybrid_tsne.model.state_dict(), f'hybrid_tsne_model_{w}.pth')

    embeddings.append(hybrid_tsne.apply(mnist_subset, "embed"))

    train_logits = hybrid_tsne.apply(mnist_subset, "classify")
    train_prediction = th.argmax(train_logits, dim=1)
    train_accuracies.append(
        th.sum(train_prediction == target_subset) / num_items
    )

    test_logits = hybrid_tsne.apply(
        mnist_data[num_items: 2 * num_items], "classify"
    )
    test_prediction = th.argmax(test_logits, dim=1)
    test_accuracies.append(
        th.sum(test_prediction == mnist.targets[num_items: 2 * num_items])
        / num_items
    )
    test_accuracies_array = np.array(test_accuracies)
    train_accuracies_array = np.array(train_accuracies)

        # Save the arrays to files
#np.save('test_accuracies.npy', test_accuracies_array)
#np.save('train_accuracies.npy', train_accuracies_array)
#th.save(tsne.training_phases[1].loss.history, "losses.pt")

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
ax.set_ylim([0, 1])
ax.legend(["train", "test"])
ax.set_title("Classification accuracy")
