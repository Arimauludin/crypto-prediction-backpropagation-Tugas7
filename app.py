from flask import Flask, render_template, request
import pandas as pd
import joblib
import io
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# 1. LOAD MODEL & SCALER (Gak perlu training lagi)
model = joblib.load('model_backpro.pkl')
scaler_X = joblib.load('scaler.pkl')

# Ambil data untuk hitung MSE & Grafik
dataset = pd.read_csv('crypto_data.csv')
X_all = dataset[['Open', 'High', 'Low', 'Volume']].values
y_true = dataset['Close'].values
X_all_scaled = scaler_X.transform(X_all)

# Hitung MSE
y_pred_all = model.predict(X_all_scaled)
mse_value = mean_squared_error(y_true, y_pred_all)

def build_plot():
    plt.figure(figsize=(6, 4))
    # MLPRegressor menyimpan loss_curve_ setelah di-load
    plt.plot(model.loss_curve_)
    plt.title('Grafik Loss (Proses Backpropagation)')
    plt.xlabel('Epoch (Iterasi)')
    plt.ylabel('Loss Value')
    plt.grid(True)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plot_url = build_plot()
    
    if request.method == 'POST':
        val_open = float(request.form['open'])
        val_high = float(request.form['high'])
        val_low = float(request.form['low'])
        val_volume = float(request.form['volume'])
        
        user_input_scaled = scaler_X.transform([[val_open, val_high, val_low, val_volume]])
        result = model.predict(user_input_scaled)
        prediction = f"{result[0]:,.2f}"
        
    return render_template('index.html', 
                           prediction=prediction, 
                           mse=f"{mse_value:,.2f}", 
                           plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)