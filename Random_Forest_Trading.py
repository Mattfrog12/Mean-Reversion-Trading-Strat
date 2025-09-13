import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.api import SimpleExpSmoothing
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


#These stocks were selected randomly
stock_tickers = ['AAPL','GOOGL','CNC','TSLA','JNJ','ADM','SYY','CAT','MRK','AAL','ORCL','INGM','GD'
                  ,'KO','TMO','BBY','NOC','NFLX','GE','HON','DHI','GEV','PNC','CMI']

#These stocks were selected based on their autocorrelation scores (less than -0.1)
#stock_tickers_mean_rev = ['ADBE','KIM','RVTY','WSM','LIN','MGM','QCOM','TPL','CNC','JNJ','AAL','INGM','BBY','DHI']


for ticker in stock_tickers:

    #grabbing data and putting it into a dataframe
    df = yf.download(ticker, start='2020-01-01')

    closing_prices = df['Close']

    #Smoothing the data
    model_1 = SimpleExpSmoothing(closing_prices)
    model_single_fit_1 = model_1.fit()

    #fitted values from exp smoothing
    weighted_cp = model_single_fit_1.fittedvalues

    model_2 = SimpleExpSmoothing(df['Volume'])
    model_single_fit_2 = model_2.fit()

    weighted_vol = model_single_fit_2.fittedvalues

    df['Weighted Close'] = weighted_cp
    df['Volume'] = weighted_vol
    df['Volatility'] = weighted_cp.rolling(window=10).std()

    #Computes RSI (14-day) at time t

    prices = weighted_cp
    delta = weighted_cp.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))


    #Removed this feature due to a low MDI score.
    #Computes Stochastic Oscillator (14-day) at time t

    # H_14 = weighted_cp.rolling(window=14).max()
    # L_14 = weighted_cp.rolling(window=14).min()
    # curr_price = weighted_cp
    #
    # df['k_value'] = 100*(curr_price - L_14)/(H_14 - L_14)


    #Computes Moving Average Convergence Divergence

    ema_12 = weighted_cp.ewm(span=12, adjust=False).mean()
    ema_26 = weighted_cp.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    #Price Rate of Change
    df['price_roc'] = ((weighted_cp - weighted_cp.shift(13)) / weighted_cp.shift(13))*100


    #Calculates OBV
    df['OBV'] = (np.sign(weighted_cp.diff()) * weighted_vol).fillna(0).cumsum()

    #Calulates Momentum over about 3 months
    momentum_3m = df.pct_change(63)
    df['Momentum'] = momentum_3m['Close']

    # Rolling window autocorrelation
    window = 20
    df['lag1_autocorr'] = df['Close'].rolling(window).apply(lambda x: x.autocorr(lag=1))


    #Shift values to not use today's data

    df['RSI'] = df['RSI'].shift(1)
    # df['k_value'] = df['k_value'].shift(1)
    df['MACD'] = df['MACD'].shift(1)
    df['Signal'] = df['Signal'].shift(1)
    df['price_roc'] = df['price_roc'].shift(1)
    df['OBV'] = df['OBV'].shift(1)
    df['Volatility'] = df['Volatility'].shift(1)
    df['Momentum'] = df['Momentum'].shift(1)
    df['lag1_autocorr'] = df['lag1_autocorr'].shift(1)



    #We need to shift our target values up by a day so that we are not using the data
    #from the day to predict the data of that day.

    df['target'] = df['Close'].shift(-1)
    df = df.dropna()



    # Random Forest Implementation Starts Here



    #Defines our feature vectors

    X_buy = df[['RSI', 'MACD', 'Signal', 'price_roc', 'OBV', 'Volatility', 'Momentum', 'lag1_autocorr']]
    X_sell = df[['RSI', 'MACD', 'Signal', 'price_roc', 'OBV', 'Volatility', 'Momentum', 'lag1_autocorr']]

    #Defines our targets (5-day forecast)

    y_buy = np.array((df['Close'].shift(-5) > df['Close']).astype(int))
    y_sell = np.array((df['Close'].shift(-5) < df['Close']).astype(int))


    X = df[['RSI', 'MACD', 'Signal', 'price_roc', 'OBV', 'Volatility', 'Momentum', 'lag1_autocorr']]
    y = (df['Close'].shift(-5) > df['Close']).astype(int)  # Buy signal target
    df = df.dropna()

    # Split data into train/test sets
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50,100,150,200],
        'max_depth': range(2,11),
        'min_samples_split': range(2,6),
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }

    # Mitigates data leak.
    tscv = TimeSeriesSplit(n_splits=5)

    # GridSearchCV setup
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',  # Could also use 'roc_auc', 'f1', etc.
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )

    # Fits the model
    grid_search.fit(X_train, y_train)

    # Best model output
    best_model = grid_search.best_estimator_
    print("Best parameters found:", grid_search.best_params_)

    # Predict on test set using best model
    y_pred = best_model.predict(X_test)

    # Store predictions and calculate strategy returns
    test_df = df.iloc[split_idx:].copy()
    test_df['market_ret'] = test_df['Close'].pct_change()
    test_df['True Signal'] = y_pred.astype(int)

    # Split data, generate models
    split_idx = int(len(df) * 0.8)
    # X_train_buy, X_test_buy = X_buy.iloc[:split_idx], X_buy.iloc[split_idx:]
    # y_train_buy, y_test_buy = y_buy[:split_idx], y_buy[split_idx:]
    #
    # model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], oob_score=True,
    # class_weight='balanced', max_depth=grid_search.best_params_['max_depth'], min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    # min_samples_split=grid_search.best_params_['min_samples_split'])
    # model.fit(X_train_buy, y_train_buy.ravel())
    # y_pred = model.predict(X_test_buy)
    #
    # X_train_sell, X_test_sell = X_sell.iloc[:split_idx], X_sell.iloc[split_idx:]
    # y_train_sell, y_test_sell = y_sell[:split_idx], y_sell[split_idx:]
    #
    # sell_model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], oob_score=True,
    # class_weight='balanced', max_depth=grid_search.best_params_['max_depth'], min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    # min_samples_split=grid_search.best_params_['min_samples_split'])
    # sell_model.fit(X_train_sell, y_train_sell.ravel())
    #
    # sell_preds = sell_model.predict(X_test_sell)



    test_df = df.iloc[split_idx:].copy()
    test_df['Market Return'] = test_df['Close'].pct_change()
    # test_df['True Signal'] = y_pred.astype(int)
    # test_df['Sell Signal'] = sell_preds.astype(int)



    test_df['cumul_market_ret'] = (1 + test_df['Market Return'].fillna(0)).cumprod()

    market_return = test_df['cumul_market_ret'][-1]

    avg_returns = {}
    values_box = []

    for j in range(50):
        # Redefine target and features
        y_buy = (df['Close'].shift(-5) > df['Close']).astype(int)
        y_sell = (df['Close'].shift(-5) < df['Close']).astype(int)
        X = df[['RSI', 'MACD', 'Signal', 'price_roc', 'OBV', 'Volatility', 'Momentum', 'lag1_autocorr']]

        split_idx = int(len(df) * 0.8)

        # Train buy model
        X_train_buy, X_test_buy = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_buy, y_test_buy = y_buy.iloc[:split_idx], y_buy.iloc[split_idx:]

        buy_model = RandomForestClassifier(
            n_estimators=grid_search.best_params_['n_estimators'],
            oob_score=True,
            class_weight='balanced',
            max_depth=grid_search.best_params_['max_depth'],
            min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
            min_samples_split=grid_search.best_params_['min_samples_split'],
            random_state=j  # vary random state for each iteration
        )
        buy_model.fit(X_train_buy, y_train_buy)
        buy_preds = buy_model.predict(X_test_buy)

        # Train sell model
        y_train_sell, y_test_sell = y_sell.iloc[:split_idx], y_sell.iloc[split_idx:]

        sell_model = RandomForestClassifier(
            n_estimators=grid_search.best_params_['n_estimators'],
            oob_score=True,
            class_weight='balanced',
            max_depth=grid_search.best_params_['max_depth'],
            min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
            min_samples_split=grid_search.best_params_['min_samples_split'],
            random_state=j
        )
        sell_model.fit(X_train_buy, y_train_sell)
        sell_preds = sell_model.predict(X_test_buy)

        # Strategy logic
        test_df = df.iloc[split_idx:].copy()
        test_df['market_ret'] = test_df['Close'].pct_change().fillna(0)

        # Apply holding logic
        position = 0  # 1 = long, -1 = short, 0 = flat
        hold_counter = 0
        positions = []

        for i in range(len(test_df)):

            # Open new position if not holding
            if buy_preds[i] == 1 and sell_preds[i] == 0:
                position = 1
            else:
                position = 0

            positions.append(position)

        test_df['Position'] = positions
        test_df['Strategy Return'] = test_df['Position'] * test_df['market_ret']
        test_df['Cumulative Strategy Return'] = (1 + test_df['Strategy Return']).cumprod()

        values_box.append(test_df['Cumulative Strategy Return'].iloc[-1])
    # Market benchmark
    market_return = test_df['Close'].pct_change().fillna(0).add(1).cumprod().iloc[-1]

    avg_strategy_return = np.mean(values_box)
    relative_perf = (avg_strategy_return - market_return) / market_return * 100

    print(f"ticker = {ticker}, strategy outperformance vs. market = {relative_perf}")

    # Mean Decrease Impurity Measure and Display
    feature_importances = buy_model.feature_importances_

    for feature, importance in zip(X.columns, feature_importances):
        print(f"{feature}: {importance:.4f}")


