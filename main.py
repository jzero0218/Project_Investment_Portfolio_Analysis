import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

tickers = [
    "VOO", 
    "0050.TW", 
    "2330.TW", 
    "2454.TW", 
    "2881.TW"] #2330 台積電 2454 聯發科 2881 富邦金 0050 元大台灣50 VOO 美國標普500

data = yf.download(
    tickers, 
    start="2020-01-01" #不放end 會抓到今天的資料
)

data.head() #了解頭五筆資料
data.shape #先了解資料結構
data.columns #先了解資料有哪些欄位
data.info()

#Data Cleaning
print(data.isnull().sum()) #檢查缺失值
print(data.isnull().sum().sum()) #檢查缺失值總數

#Close
close = data['Close']
print(close.head())
print(close.shape)
print(close.isnull().sum()) #檢查缺失值

returns = close.pct_change(fill_method = None) #計算報酬率，報酬率是今天值減去昨天的值再除以昨天的值
print(returns.head())
print(returns.describe())

#Plotting the daily returns of the selected assets
returns.plot(figsize=(12, 6), alpha=0.2, title="Daily Returns of Selected Assets")

xlabel = "Date"
ylabel = "Daily Returns"
plt.show()

#Plotting histogram of daily returns for each asset
returns.plot(kind='hist', bins=50, figsize=(12,6), alpha=0.2, legend=True)
xlabel = "Daily Returns"
ylabel = "Frequency"
plt.title("Histogram of Daily Returns for Selected Assets")
plt.show()

#去缺失值
tsmc = returns["2330.TW"].dropna() #台積電去除缺失值
voo = returns["VOO"].dropna() #美國標普500去除缺失值
etf0050 = returns["0050.TW"].dropna() #0050去除缺失值
mediantek = returns["2454.TW"].dropna() #聯發科去除缺失值
fubon = returns["2881.TW"].dropna() #富邦金去除缺失值

#檢查變異數同質
result_levene = stats.levene(tsmc, voo, etf0050, mediantek, fubon) #檢查變異數同質性，會影響後續分析的方法
print(result_levene)

#相關性
result_corr = returns.corr() #計算相關性
print(result_corr)

#共變性
result_cov = returns.cov() #計算共變性
print(result_cov)

# 台積電 + 0050投資組合設定 weight setting
weights = np.array([0.6, 0.4]) #投資率設定，台積電60%，0050 40%
cov_matrix = returns[["2330.TW", "0050.TW"]].cov()
print(cov_matrix)

#Portfolio Variance 投資組合變異數(變異P = 比重1二次方 * 變異數1 + 比重2二次方 * 變異數2 + 2 * 比重1 * 比重2 * 共變數)
portfolio_variance = weights.T @ cov_matrix @ weights
portfolio_std_dev = np.sqrt(portfolio_variance)
print("Portfolio Variance:", portfolio_variance)
print("Portfolio Standard Deviation:", portfolio_std_dev) #代表平均每天的fluctuation，越小代表越穩定

#Annualized Volatility 年化波動率
annualized_volatility = portfolio_std_dev * np.sqrt(252) #假設一年有252個交易日
print("Annualized Volatility:", annualized_volatility)

#Annualized Return 年化報酬率
mean_daily_returns = returns[["2330.TW", "0050.TW"]].mean() #每日平均報酬率
print("Mean Daily Returns:", mean_daily_returns)
protfolio_daily_return = np.sum(mean_daily_returns * weights) #投資組合每日平均報酬率
annualized_return = protfolio_daily_return * 252 #年化報酬率
print("Annualized Return:", annualized_return)

#Sharpe Ratio 夏普比率 Google 定義: 夏普比率 = (投資組合報酬率 - 無風險利率) / 投資組合波動率，用來衡量投資組合每單位波動風險，可以獲得多少超過無風險比率的報酬
#簡單來說就是把波動納進來，看看報酬率是否值得承擔風險，越高代表越值得投資。無風險利率就是你把錢放在銀行定存的利率(或者是我自己定的一個利率)，國泰世華定期儲蓄是1.72%
risk_free_rate = 0.0172 #無風險利率
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
print("Sharpe Ratio:", sharpe_ratio)

#Heatmap of Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(result_corr, annot = True, fmt = ".2f", cmap = "coolwarm")
plt.title("Correlation Heatmap of Stock Returns")
plt.savefig("images/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()