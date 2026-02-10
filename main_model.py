import torch

# from ai.gru_model import Model
# from ai.standard_model import Model
from ai.lstm_model import Model
from ai.dataset import Loader
from torch.utils.data import DataLoader, Subset
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam


def calculate_mape(predictions, targets):
    """Calculate Mean Absolute Percentage Error"""
    # Avoid division by zero by adding small epsilon or filtering zero targets
    mask = targets != 0
    if mask.sum() == 0:
        return float("inf")
    return (
        torch.mean(torch.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    )


def evaluate_model(
    model, data_loader, criterion, mae_criterion, name="", return_predictions=False
):
    """Evaluate model on a dataset and return metrics"""
    model.eval()
    total_loss = 0
    total_mae = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, target in data_loader:
            # Convert to tensors properly
            if isinstance(features, torch.Tensor):
                features = features.clone().detach().float()
            else:
                features = torch.as_tensor(features, dtype=torch.float32)

            if isinstance(target, torch.Tensor):
                target = target.clone().detach().float()
            else:
                target = torch.as_tensor(target, dtype=torch.float32)

            output = model(features)
            loss = criterion(output, target)
            mae = mae_criterion(output, target)

            total_loss += loss.item()
            total_mae += mae.item()
            all_predictions.append(output)
            all_targets.append(target)

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    mape = calculate_mape(all_predictions, all_targets)

    avg_loss = total_loss / len(data_loader)
    avg_mae = total_mae / len(data_loader)

    if name:
        print(f"\n{name} Results:")
        print(f"  Loss (MSE): {avg_loss:.4f}")
        print(f"  MAE: {avg_mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")

    if return_predictions:
        return avg_loss, avg_mae, mape.item(), all_predictions, all_targets
    return avg_loss, avg_mae, mape.item()


if __name__ == "__main__":
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

    model = Model(input_size=10)
    criterion = MSELoss()
    mae_criterion = L1Loss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    best_val_loss = float("inf")
    patience = 50  # Number of epochs to wait for improvement
    patience_counter = 0

    # Training loop
    for epoch in range(100_000):
        # Training
        model.train()
        train_loss = 0
        for features, target in train_loader:
            # Convert to tensors properly
            if isinstance(features, torch.Tensor):
                features = features.clone().detach().float()
            else:
                features = torch.as_tensor(features, dtype=torch.float32)

            if isinstance(target, torch.Tensor):
                target = target.clone().detach().float()
            else:
                target = torch.as_tensor(target, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluation on validation set
        avg_val_loss, avg_val_mae, val_mape = evaluate_model(
            model, val_loader, criterion, mae_criterion
        )

        # Evaluation on test set
        avg_test_loss, avg_test_mae, test_mape = evaluate_model(
            model, test_loader, criterion, mae_criterion
        )

        avg_train_loss = train_loss / len(train_loader)

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_mae": avg_val_mae,
                    "val_mape": val_mape,
                    "test_loss": avg_test_loss,
                    "test_mae": avg_test_mae,
                    "test_mape": test_mape,
                },
                "best_model.pth",
            )
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}, MAPE: {val_mape:.2f}% | "
                f"Test Loss: {avg_test_loss:.4f}, MAE: {avg_test_mae:.4f}, MAPE: {test_mape:.2f}% | "
                f"Best Val Loss: {best_val_loss:.4f} | Patience: {patience_counter}/{patience}"
            )

        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    print("\n" + "=" * 80)
    print("Loading best model for final evaluation...")
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Best model was saved at epoch {checkpoint['epoch']+1}")

    # Final evaluation on all datasets
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    _, _, _, train_preds, train_targets = evaluate_model(
        model,
        train_loader,
        criterion,
        mae_criterion,
        "Training Set",
        return_predictions=True,
    )
    _, _, _, val_preds, val_targets = evaluate_model(
        model,
        val_loader,
        criterion,
        mae_criterion,
        "Validation Set",
        return_predictions=True,
    )
    _, _, _, test_preds, test_targets = evaluate_model(
        model,
        test_loader,
        criterion,
        mae_criterion,
        "Test Set",
        return_predictions=True,
    )
    _, _, _, full_preds, full_targets = evaluate_model(
        model,
        full_loader,
        criterion,
        mae_criterion,
        "Full Dataset (All Data)",
        return_predictions=True,
    )

    # Print predictions
    print("\n" + "=" * 80)
    print("PREDICTIONS vs ACTUAL VALUES")
    print("=" * 80)

    datasets = [
        ("Training Set", train_preds, train_targets),
        ("Validation Set", val_preds, val_targets),
        ("Test Set", test_preds, test_targets),
        ("Full Dataset", full_preds, full_targets),
    ]

    for dataset_name, preds, targets in datasets:
        print(f"\n{dataset_name}:")
        print("-" * 80)
        print(
            f"{'Index':<8} {'Predicted':<15} {'Actual':<15} {'Error':<15} {'Error %':<10}"
        )
        print("-" * 80)

        for i in range(min(20, len(preds))):  # Show first 20 samples
            pred_val = preds[i].item()
            target_val = targets[i].item()
            error = pred_val - target_val
            error_pct = (error / target_val * 100) if target_val != 0 else float("inf")

            print(
                f"{i:<8} {pred_val:<15.4f} {target_val:<15.4f} {error:<15.4f} {error_pct:<10.2f}%"
            )

        if len(preds) > 20:
            print(f"... ({len(preds) - 20} more samples)")

        print(f"\nSummary Statistics for {dataset_name}:")
        print(f"  Total samples: {len(preds)}")
        print(f"  Mean prediction: {preds.mean().item():.4f}")
        print(f"  Mean actual: {targets.mean().item():.4f}")
        print(f"  Std prediction: {preds.std().item():.4f}")
        print(f"  Std actual: {targets.std().item():.4f}")

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
