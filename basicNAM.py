import pandas as pd
import numpy as np
import torch as th
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim


class NAM(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super(NAM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.submodules = th.nn.ModuleList()

        for i in range(input_dim):
            submodule = th.nn.Sequential()
            for l in range(num_layers):
                if l == 0:
                    submodule.add_module(f"linear_{l}", th.nn.Linear(1, hidden_dim))
                else:
                    submodule.add_module(f"linear_{l}", th.nn.Linear(hidden_dim, hidden_dim))
                submodule.add_module(f"ELU_{l}", th.nn.ELU())
                submodule.add_module(f"dropout_{l}", th.nn.Dropout(0.5))

            submodule.add_module(f"linear_{num_layers}", th.nn.Linear(hidden_dim, output_dim))
            self.submodules.append(submodule)

    def forward(self, x):
        output = th.zeros(x.shape[0], self.output_dim)

        for i in range(self.input_dim):
            output += self.submodules[i](x[:, i].unsqueeze(1))

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


class NAMClassifier:
    def __init__(self, hidden_dim=10, output_dim=1, num_layers=3):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def preprocess_dataframe(self, df):
        for column in df.columns:
            if len(df[column].unique()) == 2 and df[column].dtype == object:
                unique_labels = df[column].unique()
                mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                df[column] = df[column].map(mapping)
            elif df[column].dtype == object:
                dummies = pd.get_dummies(df[column], prefix=column)
                df = pd.concat([df, dummies], axis=1)
                df.drop(column, axis=1, inplace=True)

        return df

    def fit(self, X, y, test_size=0.2, random_state=0, lr=0.001, epochs=1000):
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y

        df = self.preprocess_dataframe(df)

        target_column = 'target'
        X = df.drop([target_column], axis=1)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        input_dim = X_train.shape[1]
        model = NAM(input_dim, self.hidden_dim, self.output_dim, self.num_layers)

        seed = 42
        th.manual_seed(seed)
        np.random.seed(seed)

        model.apply(model.init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = th.nn.BCELoss()

        for epoch in range(epochs):
            self.train(model, X_train, y_train, optimizer, loss_fn)
            test_loss, test_accuracy = self.evaluate(model, X_test, y_test, loss_fn)

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        print(f'Final test accuracy: {test_accuracy:.4f}')

    def train(self, model, X, y, optimizer, loss_fn):
        model.train()
        optimizer.zero_grad()
        X_tensor = th.tensor(X.to_numpy(), dtype=th.float32)
        y_tensor = th.tensor(y.to_numpy(), dtype=th.float32).unsqueeze(1)
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, model, X, y, loss_fn):
        model.eval()
        X_tensor = th.tensor(X.to_numpy(), dtype=th.float32)
        y_tensor = th.tensor(y.to_numpy(), dtype=th.float32).unsqueeze(1)
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        predictions = (output > 0.5).float()
        correct = (predictions == y_tensor).sum().item()
        accuracy = correct / y_tensor.shape[0]
        return loss.item(), accuracy

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained. Call 'fit' first.")

        X_tensor = th.tensor(X, dtype=th.float32)
        output = self.model(X_tensor)
        predictions = (output > 0.5).float().detach().numpy()
        return predictions.flatten()


# Example usage
#data = pd.read_csv('C:\\Users\\isabe\\Documents\\AI studies\\6.Semester\\Bachelor Thesis\\Paradime-NAMs\\Datasets\\heart.csv')  # Replace with the path to your dataset
#X = data.drop(columns=['target']).values
#y = data['target'].values

#classifier = NAMClassifier(hidden_dim=10, output_dim=1, num_layers=3)
#classifier.fit(X, y)
#predictions = classifier.predict(X)