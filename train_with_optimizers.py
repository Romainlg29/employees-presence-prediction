from torch.optim import Adam, SGD, RMSprop, AdamW

from ai.runner import Runner
from ai.lstm_model import Model as LSTMModel
from ai.gru_model import Model as GRUModel
from ai.rnn_model import Model as RNNModel
from ai.timexer_model import TimeXer


if __name__ == "__main__":
    
    # Liste des modèles à tester
    models = [
        (lambda i: LSTMModel(i), "LSTM"),
        (lambda i: GRUModel(i), "GRU"),
        (lambda i: RNNModel(i), "RNN"),
        (lambda i: TimeXer(i), "TimeXer"),
    ]
    
    # Liste des optimizers à tester avec leurs learning rates adaptés
    optimizers = [
        (Adam, "Adam", 0.001),
        (AdamW, "AdamW", 0.001),
        (SGD, "SGD", 0.00001),
        (RMSprop, "RMSprop", 0.001),
    ]
    
    all_results = {}
    
    print("\n" + "="*70)
    print("Test des optimizers sur tous les modèles")
    print("="*70 + "\n")
    
    for model_func, model_name in models:
        print(f"\n{'#'*70}")
        print(f"# Modèle: {model_name}")
        print(f"{'#'*70}\n")
        
        model_results = []
        
        for opt_class, opt_name, lr in optimizers:
            print(f"→ Test avec {opt_name} (lr={lr})...")
            
            runner = Runner(
                path="df_venues_model.py",
                model=model_func,
                name=f"{model_name}_{opt_name}_lr{lr}",
                optimizer_class=opt_class,
                learning_rate=lr
            )
            
            evaluation_metrics, _, predictions_targets = runner.run(plot=False)
            
            # Calculate RMSE from MSE (loss)
            rmse = evaluation_metrics[0] ** 0.5
            
            model_results.append({
                'optimizer': opt_name,
                'lr': lr,
                'mse': evaluation_metrics[0],
                'mae': evaluation_metrics[1],
                'rmse': rmse,
                'mape': evaluation_metrics[2]
            })
            
            print(f"  ✓ MAE: {evaluation_metrics[1]:.4f} | RMSE: {rmse:.4f} | MAPE: {evaluation_metrics[2]:.2f}%\n")
        
        all_results[model_name] = model_results
    
    # Résumé global
    print("\n" + "="*70)
    print("RÉSUMÉ GLOBAL")
    print("="*70)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print("-" * 70)
        for r in results:
            print(f"  {r['optimizer']:10} (lr={r['lr']}) - MAE: {r['mae']:.4f} | RMSE: {r['rmse']:.4f} | MAPE: {r['mape']:.2f}%")
        
        best_mae = min(results, key=lambda x: x['mae'])
        print(f"  → Meilleur: {best_mae['optimizer']} (MAE: {best_mae['mae']:.4f})")
    
    # Meilleur optimizer global
    print("\n" + "="*70)
    print("MEILLEUR OPTIMIZER PAR MODÈLE")
    print("="*70)
    
    for model_name, results in all_results.items():
        best = min(results, key=lambda x: x['mae'])
        print(f"{model_name:10} → {best['optimizer']:10} (lr={best['lr']}) - MAE: {best['mae']:.4f} | RMSE: {best['rmse']:.4f}")