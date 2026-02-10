from ai.base import Base
from ai.dataset import Loader

# from ai.gru_model import Model
# from ai.standard_model import Model
from ai.lstm_model import Model

if __name__ == "__main__":

    # Load the dataset
    dataset = Loader("df_venues_processed.csv")

    # Wrapper
    base = Base(dataset, model=lambda i: Model(i), name="LSTM")

    # Prepare the data and model
    base.prepare()

    # Train the model
    epoch = base.fit(epochs=250)
    print(f"Training completed in {epoch} epochs.")

    # Evaluate on the validation set
    evaluation_loss, evaluation_mae, evaluation_mape, _, _ = base.evaluate()

    print(f"\nFinal Evaluation on Validation Set:")
    print(f"  Loss (MSE): {evaluation_loss:.4f}")
    print(f"  MAE: {evaluation_mae:.4f}")
    print(f"  MAPE: {evaluation_mape:.2f}%")

    # Evaluate on the test set
    test_loss, test_mae, test_mape, _, _ = base.test()

    print(f"\nFinal Evaluation on Test Set:")
    print(f"  Loss (MSE): {test_loss:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
