# uv run python -m app.main

from pathlib import Path

from app.metrics import calculate_metrics
from app.neural_network import NeuralNetwork
from app.preprocessing import TitanicPreprocessor


def main():
    train_path = Path("data/train.csv")

    if not train_path.exists():
        print(f"ERR: File not found: {train_path}")
        return

    print("Trenowanie modeu...\n")

    preprocessor = TitanicPreprocessor()
    X_train, y_train, X_val, y_val = preprocessor.load_and_prepare_data(train_path, val_split=0.2)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Features: {X_train.shape[1]}\n")

    model = NeuralNetwork([X_train.shape[1], 8, 1], learning_rate=0.03, seed=42)

    #najlepsze wartości, jakie znalazłem
    model.train(X_train, y_train, epochs=200, batch_size=16)

    print("\nEvaluation:")
    y_val_pred = model.predict(X_val, threshold=0.5)
    metrics = calculate_metrics(y_val, y_val_pred)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()