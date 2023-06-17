import pandas as pd
import numpy as np
import torch as th
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim

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
            submodule.add_module(f"linear_{num_layers}", th.nn.Linear(hidden_dim, output_dim))
            self.submodules.append(submodule)

    def forward(self, x):
        output = th.zeros(x.shape[0], self.output_dim)

        for i in range(self.input_dim):
            output[:, i] = self.submodules[i](x[:, i].unsqueeze(1)).squeeze()

        return th.sigmoid(output)

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

class twoNAMClassifier:
    def __init__(self, hidden_dim, output_dim=2, num_layers=1):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def fit(self, X, y, lr=0.001, epochs=1000):
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
        df['target'] = y

        df = df.dropna()
        input_dim = len(df.columns) - 1

        target_column = 'target'
        X = df.drop([target_column], axis=1)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        input_ranges = np.zeros((X.shape[1], 2))
        for i, col in enumerate(X.columns):
            input_ranges[i, 0] = X[col].min()
            input_ranges[i, 1] = X[col].max()

        model = twoNAM(input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim, num_layers=self.num_layers)

        seed = 42
        th.manual_seed(seed)
        np.random.seed(seed)

        model.apply(model.init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = th.nn.BCELoss()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            X_tensor = th.tensor(X_train.values, dtype=th.float32)
            y_tensor = th.tensor(y_train.values, dtype=th.float32).unsqueeze(1)
            output = model(X_tensor)
            loss = loss_fn(output, y_tensor)
            loss.backward()
            optimizer.step()

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def predict(self, X):
        model = self.model.eval()
        X_tensor = th.tensor(X, dtype=th.float32)
        output = model(X_tensor)
        predictions = (output > 0.5).float()
        return predictions.detach().numpy()