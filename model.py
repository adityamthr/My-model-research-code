import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# here, i am just doing calculations for realized vols to be used in our dataframe, to train data on.
def calc_realized_vol(prices, window=20, annual_factor=np.sqrt(252*375)):
    returns = np.diff(np.log(prices))
    vols = [
        np.std(returns[i-window+1:i+1]) * annual_factor
        for i in range(window-1, len(returns))
    ]
    return np.array(vols)

# Here master Aditya M is doing Feature Engineering --
def make_features(vols, prices, iv, window=20):
    X = []
    for i in range(window, len(vols)):
        vol_window = vols[i-window:i]
        current_vol = vols[i-1]
        # Realized vol features
        rv_mom = np.mean(vol_window[-5:]) - np.mean(vol_window[:-5])
        rv_mr = current_vol - np.mean(vol_window)
        changes = np.diff(vol_window)
        rv_pers = np.corrcoef(changes[:-1], changes[1:])[0,1] if len(changes)>1 else 0
        rv_pers = 0 if np.isnan(rv_pers) else rv_pers
        rv_regime = current_vol / np.median(vol_window)
        # Price momentum
        pr_window = prices[i-window:i]
        pr_mom = np.mean(np.diff(np.log(pr_window))[-5:]) if len(pr_window)>5 else 0
        # Time of day
        tod = (i % 375)/375

        # IV features (aligned with vols: vols start at index window-1)
        iv_idx = i-1  # latest IV corresponds to that vol
        iv_atm = iv[iv_idx]
        # IV trend: change over last 5
        iv_trend = iv[iv_idx] - iv[iv_idx-5] if iv_idx>=5 else 0
        # IV-RV spread
        iv_spread = iv[iv_idx] - current_vol

        X.append([
            current_vol, rv_mom, rv_mr, rv_pers, rv_regime,
            pr_mom, tod, iv_atm, iv_trend, iv_spread
        ])
    return np.array(X)

# here we make targets for vols
def make_targets(vols, horizon=5):
    return vols[horizon:]

# -- Model Class --
class VolPredictor:
    def __init__(self, window=20, horizon=5):
        kern = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kern, alpha=1e-6,
                                           n_restarts_optimizer=5, random_state=42)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.window, self.horizon = window, horizon

    def fit(self, prices, iv):
        vols = calc_realized_vol(prices, self.window)
        X = make_features(vols, prices, iv, self.window)
        y = make_targets(vols, self.horizon)
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1,1)).ravel()
        self.gp.fit(Xs, ys)

    def predict(self, recent_prices, recent_iv):
        vols = calc_realized_vol(recent_prices, self.window)
        if len(vols) < self.window:
            raise ValueError("Not enough data to form features")
        X = make_features(vols, recent_prices, recent_iv, self.window)[-1:]
        Xs = self.scaler_X.transform(X)
        y_pred_s, y_std_s = self.gp.predict(Xs, return_std=True)
        mean = self.scaler_y.inverse_transform(y_pred_s.reshape(-1,1))[0,0]
        std = y_std_s[0] * self.scaler_y.scale_[0]
        return mean, std

if __name__ == "__main__":
    # Load OHLCV crypto data (e.g., BTC/ETH)
    df = pd.read_csv("crypto_ohlcv.csv") # here we have close and iv columns in this file included in our df

    # Extract price and IV series
    prices = df['close'].values
    iv = df['iv'].values  # gotcha take this beyaaaaccc... from options data

    model = VolPredictor(window=25, horizon=10)
    model.fit(prices, iv)

    recent_p = prices[-100:]
    recent_iv = iv[-100:]
    pred, uncert = model.predict(recent_p, recent_iv)
    print(f"Predicted vol: {pred:.2%}, Uncertainty ±{uncert:.2%}")
