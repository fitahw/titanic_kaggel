import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=67):
        if len(layer_sizes) != 3:
            raise ValueError("Błąd argumentu do NeuralNetwork, powinno być: [input_size, hidden_size, output_size].")

        input_size, hidden_size, output_size = layer_sizes

        np.random.seed(seed)
        #o ile * zmieniają się weighty
        self.learning_rate = learning_rate

        #losowa inicjalizacja losowymi wartościami
        self.weights = [
            np.random.randn(input_size, hidden_size) * 0.1,
            np.random.randn(hidden_size, output_size) * 0.1,
        ]
        #biasy startują od 0
        self.biases = [
            np.zeros((1, hidden_size)),
            np.zeros((1, output_size)),
        ]
        #wypełniane podczas treningu
        self.loss_history = []

    #ostatni krok podczas forward, zmienia wyniki w zakres 0-1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # relu, x jeśli x > 0, 0 jeśli x <= 0
    def relu(self, x):
        return np.maximum(0, x)

    # der. relu, 1 jeśli x > 0, 0 jeśli x <= 0
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    # przejście w prawo ->
    def forward(self, X):
        # wejście do pierwszej warstwy hidden = XW1 + b1
        self.hidden_input = np.dot(X, self.weights[0]) + self.biases[0]

        #usunięcie wartosci ujemnych aby zamendować liniowość
        # z relu, x jeśli x > 0, 0 jeśli x <= 0
        self.hidden_output = self.relu(self.hidden_input)

        #koniec = zrelowane dane * weighty + biasy
        output_input = np.dot(self.hidden_output, self.weights[1]) + self.biases[1]
        #return sigmoidem, z racji że output to 0 lub 1
        return self.sigmoid(output_input)

    # X, y takie same - nowy tylko output z forward co przejście
    def backward(self, X, y, output):
        #liczba próbek w batchu
        m = X.shape[0]
        #zmiana y na kolumnę (-1 = zostaw jak jest, 1 = 1 kolumna)
        y = y.reshape(-1, 1) 

        #przewidywana z forward - prawdziwa wartość
        output_error = output - y

        # hidden error = obliczony wyżej error * weighty W2
        hidden_error = np.dot(output_error, self.weights[1].T)

        # wracając mnożymy przez:
        # 1 jeśli neuron był aktywny (błąd większy od 0),
        # 0 jeśli nie (wtedy neuron się nie uczy)
        hidden_error *= self.relu_derivative(self.hidden_input)

        # nowe weighty 2 = średnia outputu z hidden warstwy * drugi error 
        dW2 = np.dot(self.hidden_output.T, output_error) / m
        #nowy bias 2 = średnia z sumy all output error
        db2 = np.sum(output_error, axis=0, keepdims=True) / m

        #nowe weighty 1 = średnia z wartosci wtórnych X * pierwszy error
        dW1 = np.dot(X.T, hidden_error) / m
        #nowy bias 2 = średnia z sumy all hidden error
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / m

        #nowe weighty = powyższe * learning rate
        self.weights[1] -= self.learning_rate * dW2
        self.biases[1] -= self.learning_rate * db2
        self.weights[0] -= self.learning_rate * dW1
        self.biases[0] -= self.learning_rate * db1

    def train(self, X_train, y_train, epochs=100, batch_size=16):
        #liczba próbek w batchu
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            #shufflowanie danych
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            #dla kazdego batcha
            for i in range(0, n_samples, batch_size):
                #branie batcha z x i y
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                # forward, backward
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

            #forward, 
            train_output = self.forward(X_train)
            #oblicza accuracy, jak bardzo sieć się myli
            train_loss = self._binary_cross_entropy(y_train, train_output)
            self.loss_history.append(train_loss)

            #print co 20 epochów
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")

    #poza treningiem, predict na danych
    # jeśli więcej niż 50% szansy na przeżycie = pasażer przeżył.
    def predict(self, X, threshold=0.5):
        #po treningu sam forward daje nam wynik
        output = self.forward(X)
        return (output >= threshold).astype(int).flatten()

    #zwraca gołe probabilities
    def predict_proba(self, X):
        # flatten robi [0, 1, 0] z:
        # [0],
        # [1],
        # [0]
        return self.forward(X).flatten()

    def _binary_cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred.flatten(), 1e-9, 1 - 1e-9)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
