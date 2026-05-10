import numpy as np
import pandas as pd

from app.metrics import calculate_metrics
from app.neural_network import NeuralNetwork
from app.preprocessing import TitanicPreprocessor


def test_network_initialization():
    nn = NeuralNetwork([10, 5, 1])
    assert len(nn.weights) == 2
    assert len(nn.biases) == 2
    assert nn.weights[0].shape == (10, 5)
    assert nn.weights[1].shape == (5, 1)


def test_sigmoid_activation():
    nn = NeuralNetwork([10, 5, 1])
    x = np.array([0.0, 1.0, -1.0])
    result = nn.sigmoid(x)
    assert np.allclose(result[0], 0.5)
    assert 0 < result[1] < 1
    assert 0 < result[2] < 1


def test_relu_activation():
    nn = NeuralNetwork([10, 5, 1])
    x = np.array([1.0, -1.0, 0.0, 2.0])
    result = nn.relu(x)
    assert np.allclose(result, [1.0, 0.0, 0.0, 2.0])


def test_forward_pass():
    np.random.seed(42)
    nn = NeuralNetwork([5, 3, 1], seed=42)
    X = np.random.randn(10, 5)
    output = nn.forward(X)
    assert output.shape == (10, 1)
    assert np.all((output >= 0) & (output <= 1))


def test_predict():
    np.random.seed(42)
    nn = NeuralNetwork([5, 3, 1], seed=42)
    X = np.random.randn(10, 5)
    predictions = nn.predict(X, threshold=0.5)
    assert predictions.shape == (10,)
    assert np.all((predictions == 0) | (predictions == 1))


def test_predict_proba():
    np.random.seed(42)
    nn = NeuralNetwork([5, 3, 1], seed=42)
    X = np.random.randn(10, 5)
    probas = nn.predict_proba(X)
    assert probas.shape == (10,)
    assert np.all((probas >= 0) & (probas <= 1))


def test_training():
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)

    nn = NeuralNetwork([10, 8, 1], learning_rate=0.01, seed=42)
    nn.train(X, y, epochs=10, batch_size=16)

    assert len(nn.loss_history) == 10
    assert all(isinstance(loss, (float, np.floating)) for loss in nn.loss_history)


def test_metrics_calculation():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["tp"] == 3
    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert np.isclose(metrics["accuracy"], 0.75)


def test_preprocessing_drops_target_and_keeps_dummy_columns(tmp_path):
    train_data = pd.DataFrame(
        {
            "Survived": [0, 1, 1, 0, 1, 0, 0, 1],
            "Pclass": [3, 1, 2, 3, 1, 2, 3, 1],
            "Sex": ["male", "female", "female", "male", "female", "male", "male", "female"],
            "Age": [22, 38, np.nan, 35, 28, 18, np.nan, 45],
            "SibSp": [1, 1, 0, 0, 1, 0, 0, 1],
            "Parch": [0, 0, 0, 0, 1, 0, 0, 1],
            "Fare": [7.25, 71.28, 8.05, 8.05, 53.1, 13.0, 7.75, 83.15],
            "Cabin": [np.nan, "C85", np.nan, np.nan, "C123", np.nan, np.nan, "B42"],
            "Embarked": ["S", "C", "S", "S", "S", "Q", "Q", "C"],
        }
    )

    train_path = tmp_path / "train.csv"
    train_data.to_csv(train_path, index=False)

    preprocessor = TitanicPreprocessor(seed=42)
    X_train, _, X_val, _ = preprocessor.load_and_prepare_data(train_path)

    assert "Survived" not in preprocessor.feature_cols
    assert "Cabin" not in preprocessor.feature_cols
    assert "Room" in preprocessor.feature_cols
    assert any(col.startswith("Deck_") for col in preprocessor.feature_cols)
    assert any(col.startswith("Sex_") for col in preprocessor.feature_cols)
    assert X_train.shape[1] == X_val.shape[1]
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()
