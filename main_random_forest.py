import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ai.lstm_model import Model
from ai.dataset import Loader
from torch.utils.data import DataLoader, Subset


def calculate_mape(predictions, targets):
    """Calculate Mean Absolute Percentage Error"""
    mask = targets != 0
    if mask.sum() == 0:
        return float("inf")

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    return np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100


def extract_data_from_loader(data_loader):
    """Extract all features and targets from a DataLoader"""
    all_features = []
    all_targets = []

    for features, target in data_loader:
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()

        all_features.append(features)
        all_targets.append(target)

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)

    return X, y


def evaluate_random_forest(model, X, y, name=""):
    """Evaluate Random Forest model"""
    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    mape = calculate_mape(predictions, y)

    if name:
        print(f"\n{name} Results:")
        print(f"  Loss (MSE): {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")

    return mse, mae, mape, predictions


def evaluate_neural_network(model, data_loader, name=""):
    """Evaluate Neural Network model"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, target in data_loader:
            if isinstance(features, torch.Tensor):
                features = features.clone().detach().float()
            else:
                features = torch.as_tensor(features, dtype=torch.float32)

            if isinstance(target, torch.Tensor):
                target = target.clone().detach().float()
            else:
                target = torch.as_tensor(target, dtype=torch.float32)

            output = model(features)
            all_predictions.append(output)
            all_targets.append(target)

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    mape = calculate_mape(predictions, targets)

    if name:
        print(f"\n{name} Results:")
        print(f"  Loss (MSE): {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")

    return mse, mae, mape, predictions.numpy(), targets.numpy()


if __name__ == "__main__":
    print("=" * 80)
    print("MODEL COMPARISON: Random Forest vs Neural Network (LSTM)")
    print("=" * 80)

    # Load dataset
    dataset = Loader("df_venues_processed.csv").encode()

    # Train/val/test split - temporal order preserved (70/15/15 split)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))

    train_dataset = Subset(dataset, range(train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Extract data for Random Forest
    print("\nExtracting data for Random Forest...")
    X_train, y_train = extract_data_from_loader(train_loader)
    X_val, y_val = extract_data_from_loader(val_loader)
    X_test, y_test = extract_data_from_loader(test_loader)
    X_full, y_full = extract_data_from_loader(full_loader)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Total samples: {len(X_full)}")

    # Train Random Forest
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST REGRESSOR")
    print("=" * 80)

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("\nTraining Random Forest...")
    rf_model.fit(X_train, y_train)
    print("Training completed!")

    # Evaluate Random Forest
    print("\n" + "=" * 80)
    print("RANDOM FOREST EVALUATION")
    print("=" * 80)

    rf_train_mse, rf_train_mae, rf_train_mape, rf_train_preds = evaluate_random_forest(
        rf_model, X_train, y_train, "Random Forest - Training Set"
    )
    rf_val_mse, rf_val_mae, rf_val_mape, rf_val_preds = evaluate_random_forest(
        rf_model, X_val, y_val, "Random Forest - Validation Set"
    )
    rf_test_mse, rf_test_mae, rf_test_mape, rf_test_preds = evaluate_random_forest(
        rf_model, X_test, y_test, "Random Forest - Test Set"
    )
    rf_full_mse, rf_full_mae, rf_full_mape, rf_full_preds = evaluate_random_forest(
        rf_model, X_full, y_full, "Random Forest - Full Dataset"
    )

    # Load best Neural Network model
    print("\n" + "=" * 80)
    print("LOADING NEURAL NETWORK MODEL (LSTM)")
    print("=" * 80)

    nn_model = Model(input_size=10)
    try:
        checkpoint = torch.load("best_model.pth")
        nn_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    except FileNotFoundError:
        print(
            "Warning: best_model.pth not found. Please train the neural network first!"
        )
        print("Run main.py to train the model.")
        exit(1)

    # Evaluate Neural Network
    print("\n" + "=" * 80)
    print("NEURAL NETWORK EVALUATION")
    print("=" * 80)

    nn_train_mse, nn_train_mae, nn_train_mape, nn_train_preds, nn_train_targets = (
        evaluate_neural_network(nn_model, train_loader, "Neural Network - Training Set")
    )
    nn_val_mse, nn_val_mae, nn_val_mape, nn_val_preds, nn_val_targets = (
        evaluate_neural_network(nn_model, val_loader, "Neural Network - Validation Set")
    )
    nn_test_mse, nn_test_mae, nn_test_mape, nn_test_preds, nn_test_targets = (
        evaluate_neural_network(nn_model, test_loader, "Neural Network - Test Set")
    )
    nn_full_mse, nn_full_mae, nn_full_mape, nn_full_preds, nn_full_targets = (
        evaluate_neural_network(nn_model, full_loader, "Neural Network - Full Dataset")
    )

    # Comparison Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    datasets_info = [
        (
            "Training Set",
            (rf_train_mse, rf_train_mae, rf_train_mape),
            (nn_train_mse, nn_train_mae, nn_train_mape),
        ),
        (
            "Validation Set",
            (rf_val_mse, rf_val_mae, rf_val_mape),
            (nn_val_mse, nn_val_mae, nn_val_mape),
        ),
        (
            "Test Set",
            (rf_test_mse, rf_test_mae, rf_test_mape),
            (nn_test_mse, nn_test_mae, nn_test_mape),
        ),
        (
            "Full Dataset",
            (rf_full_mse, rf_full_mae, rf_full_mape),
            (nn_full_mse, nn_full_mae, nn_full_mape),
        ),
    ]

    for (
        dataset_name,
        (rf_mse, rf_mae, rf_mape),
        (nn_mse, nn_mae, nn_mape),
    ) in datasets_info:
        print(f"\n{dataset_name}:")
        print("-" * 80)
        print(
            f"{'Metric':<20} {'Random Forest':<20} {'Neural Network':<20} {'Winner':<20}"
        )
        print("-" * 80)

        # MSE comparison
        mse_winner = "Random Forest" if rf_mse < nn_mse else "Neural Network"
        mse_diff = abs(rf_mse - nn_mse) / max(rf_mse, nn_mse) * 100
        print(
            f"{'MSE':<20} {rf_mse:<20.4f} {nn_mse:<20.4f} {mse_winner:<20} (Δ {mse_diff:.2f}%)"
        )

        # MAE comparison
        mae_winner = "Random Forest" if rf_mae < nn_mae else "Neural Network"
        mae_diff = abs(rf_mae - nn_mae) / max(rf_mae, nn_mae) * 100
        print(
            f"{'MAE':<20} {rf_mae:<20.4f} {nn_mae:<20.4f} {mae_winner:<20} (Δ {mae_diff:.2f}%)"
        )

        # MAPE comparison
        mape_winner = "Random Forest" if rf_mape < nn_mape else "Neural Network"
        mape_diff = abs(rf_mape - nn_mape) / max(rf_mape, nn_mape) * 100
        print(
            f"{'MAPE (%)':<20} {rf_mape:<20.2f} {nn_mape:<20.2f} {mape_winner:<20} (Δ {mape_diff:.2f}%)"
        )

    # Overall winner
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    # Calculate average performance across all metrics on test set
    rf_avg_score = (rf_test_mse + rf_test_mae + rf_test_mape) / 3
    nn_avg_score = (nn_test_mse + nn_test_mae + nn_test_mape) / 3

    print(f"\nBased on Test Set Performance:")
    print(f"  Random Forest Average Score: {rf_avg_score:.4f}")
    print(f"  Neural Network Average Score: {nn_avg_score:.4f}")

    if rf_avg_score < nn_avg_score:
        improvement = (nn_avg_score - rf_avg_score) / nn_avg_score * 100
        print(f"\n🏆 Random Forest performs better by {improvement:.2f}%")
    else:
        improvement = (rf_avg_score - nn_avg_score) / rf_avg_score * 100
        print(f"\n🏆 Neural Network performs better by {improvement:.2f}%")

    # Sample predictions comparison
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS COMPARISON (First 10 Test Samples)")
    print("=" * 80)
    print(
        f"{'Index':<8} {'Actual':<15} {'RF Pred':<15} {'NN Pred':<15} {'RF Error':<15} {'NN Error':<15}"
    )
    print("-" * 95)

    for i in range(min(10, len(y_test))):
        actual = y_test[i]
        rf_pred = rf_test_preds[i]
        nn_pred = nn_test_preds[i]
        rf_error = abs(actual - rf_pred)
        nn_error = abs(actual - nn_pred)

        print(
            f"{i:<8} {actual:<15.4f} {rf_pred:<15.4f} {nn_pred:<15.4f} {rf_error:<15.4f} {nn_error:<15.4f}"
        )

    print("\n" + "=" * 80)
    print("Comparison completed!")
    print("=" * 80)
