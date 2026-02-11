import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ai.dataset import Loader
from torch.utils.data import DataLoader, Subset

from ai.base import Base 


class RandomForestModel():
    def __init__(self):
        
        self._rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

    def _calculate_mape(self, predictions, targets):
        """Calculate Mean Absolute Percentage Error"""
        mask = targets != 0
        if mask.sum() == 0:
            return float("inf")

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        return np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100


    def _extract_data_from_loader(self, data_loader):
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

    def _evaluate(self, model, X, y, name=""):
        """Evaluate Random Forest model"""
        predictions = model.predict(X)

        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        mape = self._calculate_mape(predictions, y)

        if name:
            print(f"\n{name} Results:")
            print(f"  Loss (MSE): {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")

        return mse, mae, mape, predictions
    
    def _extract_data_for_random_forest(self, dataset):
         # Train/val/test split - temporal order preserved (70/15/15 split)
        train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32, shuffle=False)
        full_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Extract data for Random Forest
        print("\nExtracting data for Random Forest...")
        X_train, y_train = self._extract_data_from_loader(train_loader)
        X_val, y_val = self._extract_data_from_loader(val_loader)
        X_test, y_test = self._extract_data_from_loader(test_loader)
        X_full, y_full = self._extract_data_from_loader(full_loader)

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Total samples: {len(X_full)}")

        return X_train, y_train, X_val, y_val, X_test, y_test, X_full, y_full



    def run_random_forest(self): 
        print("=" * 80)
        print("Random Forest")
        print("=" * 80)

        # Load dataset
        dataset = Loader("df_venues_processed.csv").encode()

        # Extract data for Random Forest
        X_train, y_train, X_val, y_val, X_test, y_test, X_full, y_full = self._extract_data_for_random_forest(dataset)
        
        # Train Random Forest
        print("\n" + "=" * 80)
        print("TRAINING RANDOM FOREST REGRESSOR")
        print("=" * 80)
        self._rf_model.fit(X_train, y_train)
        print("Training completed!")

        # Evaluate Random Forest
        print("\n" + "=" * 80)
        print("RANDOM FOREST EVALUATION")
        print("=" * 80)

        rf_train_mse, rf_train_mae, rf_train_mape, rf_train_preds = self._evaluate(
            self._rf_model, X_train, y_train, "Random Forest - Training Set"
        )
        rf_val_mse, rf_val_mae, rf_val_mape, rf_val_preds = self._evaluate(
            self._rf_model, X_val, y_val, "Random Forest - Validation Set"
        )
        rf_test_mse, rf_test_mae, rf_test_mape, rf_test_preds = self._evaluate(
            self._rf_model, X_test, y_test, "Random Forest - Test Set"
        )
        rf_full_mse, rf_full_mae, rf_full_mape, rf_full_preds = self._evaluate(
            self._rf_model, X_full, y_full, "Random Forest - Full Dataset"
        )

        result = { 
            "train": (rf_train_mse, rf_train_mae, rf_train_mape),
            "val": (rf_val_mse, rf_val_mae, rf_val_mape),
            "test": (rf_test_mse, rf_test_mae, rf_test_mape),
            "full": (rf_full_mse, rf_full_mae, rf_full_mape),
        }, {
            "train": rf_train_preds,
            "val": rf_val_preds,
            "test": rf_test_preds,
            "full": rf_full_preds,
        }, {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
            "full": (X_full, y_full),
        }
        return result


    def plot_rf(self, metrics, predictions, data):
        """
        Evaluate and display Random Forest performance with visualizations
        """
        print("=" * 80)
        print("RANDOM FOREST - EVALUATION & VISUALIZATION")
        print("=" * 80)
        
        # Create a Base instance for visualization
        dataset = Loader("df_venues_processed.csv").encode()
        from ai.lstm_model import Model as LSTMModel
        base_visualizer = Base(
            dataset=dataset,
            model=lambda input_size: LSTMModel(input_size=input_size),
            name="Visualizer"
        )
        
        # Display visualizations
        print("\n" + "=" * 80)
        print("VISUALIZATIONS")
        print("=" * 80)
        
        # Test Set Analysis
        print("\n[1] Test Set Predictions vs Actual Values...")
        base_visualizer.plot_predictions(
            predictions["test"], 
            data["test"][1],
            title="Random Forest - Test Set: Predictions vs Actual Values"
        )
        
        # Validation Set Analysis
        print("\n[2] Validation Set Predictions vs Actual Values...")
        base_visualizer.plot_predictions(
            predictions["val"], 
            data["val"][1],
            title="Random Forest - Validation Set: Predictions vs Actual Values"
        )
        
        # Training Set Analysis
        print("\n[3] Training Set Predictions vs Actual Values...")
        base_visualizer.plot_predictions(
            predictions["train"], 
            data["train"][1],
            title="Random Forest - Training Set: Predictions vs Actual Values"
        )
        
        # Summary
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        for dataset_name in ["train", "val", "test"]:
            mse, mae, mape = metrics[dataset_name]
            print(f"\n{dataset_name.upper()} SET:")
            print(f"  MSE:  {mse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80) 
        
