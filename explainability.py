import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

def explain_model_decisions():
    print("[XAI] Loading Model and Data...")
    
    model = xgb.Booster()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgb_model.json")
    model.load_model(model_path)
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "btc_usdt_1h_features.csv")
    df = pd.read_csv(data_path)
    
    feature_cols = [
        "return_1h", "return_4h", "log_return", "hl_spread", 
        "co_spread", "vol_change", "volatility_10", "rsi_14_norm", 
        "macd_line_pct", "macd_signal_pct", "macd_hist_pct", 
        "ema_20_dist", "ema_50_dist", "ema_cross_20_50", "atr_pct"
    ]
    X = df[feature_cols]
    
    print("[XAI] Calculating SHAP values (This might take a few seconds)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print("\n📊 Global Feature Importance (Top 5 Features driving the AI):")
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    
    for index, row in feature_importance.head(5).iterrows():
        print(f" - {row['col_name']}: {row['feature_importance_vals']:.4f}")
        
    latest_candle = X.iloc[-1]
    latest_shap = shap_values[-1]
    
    print("\n🔍 Deep Dive: AI Decision for the LATEST Hour")
    print("------------------------------------------------")
    print("Top 3 reasons PUSHING the price UP (Bullish):")
    bullish_factors = [(feature_cols[i], latest_shap[i]) for i in range(len(feature_cols)) if latest_shap[i] > 0]
    bullish_factors.sort(key=lambda x: x[1], reverse=True)
    for feat, val in bullish_factors[:3]:
        print(f" [+] {feat} (Impact: +{val:.4f})")
        
    print("\nTop 3 reasons PUSHING the price DOWN (Bearish):")
    bearish_factors = [(feature_cols[i], latest_shap[i]) for i in range(len(feature_cols)) if latest_shap[i] < 0]
    bearish_factors.sort(key=lambda x: x[1])
    for feat, val in bearish_factors[:3]:
        print(f" [-] {feat} (Impact: {val:.4f})")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    shap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shap_summary.png")
    plt.savefig(shap_path, bbox_inches='tight', dpi=300)
    print("\n💾 Visual SHAP summary saved as 'shap_summary.png' for your Portfolio!")

if __name__ == "__main__":
    explain_model_decisions()