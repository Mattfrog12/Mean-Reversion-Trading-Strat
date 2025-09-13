import pandas as pd
import yfinance as yf

tickers = ['AAPL','GOOGL','CNC','TSLA','JNJ','ADM','SYY','CAT','MRK','AAL','ORCL','INGM','GD'
            ,'KO','TMO','BBY','NOC','NFLX','GE','HON','DHI','GEV','PNC','CMI','ABBV','ADBE',
           'AMT','CAH','CCI','FDS','HWM','KIM','MSFT','PCAR','RVTY','TXN','TDG','WSM','GWW',
           'CPT','CFG','DASH','FAST','HPQ','JKHY','LIN','MGM','ORLY','PWR','QCOM','SNA','TPL']

tickers_corr = {}

for ticker in tickers:
    df = yf.download(tickers, start='2025-01-01', end='2025-08-01')['Close']

    returns = df[f'{ticker}'].pct_change().dropna()
    autocorr = returns.autocorr(lag=1)
    tickers_corr[ticker] = autocorr

print(tickers_corr)

