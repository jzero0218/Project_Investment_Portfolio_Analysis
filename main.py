import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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
# weights = np.array([0.6, 0.4]) #投資率設定，台積電60%，0050 40%
portfolio_weights = np.array([ #投資率設定
    [1, 0],
    [0.75, 0.25],
    [0.5, 0.5],
    [0.25, 0.75],
    [0, 1]
    ])
cov_matrix = returns[["2330.TW", "0050.TW"]].cov()
print(cov_matrix)

#Portfolio Variance 投資組合變異數(變異P = 比重1二次方 * 變異數1 + 比重2二次方 * 變異數2 + 2 * 比重1 * 比重2 * 共變數)
# portfolio_variance = weights.T @ cov_matrix @ weights
# portfolio_std_dev = np.sqrt(portfolio_variance)
# print("Portfolio Variance:", portfolio_variance)
# print("Portfolio Standard Deviation:", portfolio_std_dev) #代表平均每天的fluctuation，越小代表越穩定

# #Annualized Volatility 年化波動率
# annualized_volatility = portfolio_std_dev * np.sqrt(252) #假設一年有252個交易日
# print("Annualized Volatility:", annualized_volatility)

#Annualized Return 年化報酬率
# mean_daily_returns = returns[["2330.TW", "0050.TW"]].mean() #每日平均報酬率
# print("Mean Daily Returns:", mean_daily_returns)
# protfolio_daily_return = np.sum(mean_daily_returns * weights, axis = 1) #投資組合每日平均報酬率，axis = 1 代表橫向相加，axis = 0 代表縱向相加，不加會全部加一起，只會有一個數
# annualized_return = protfolio_daily_return * 252 #年化報酬率
# print("Annualized Return:", annualized_return)

#Sharpe Ratio 夏普比率 Google 定義: 夏普比率 = (投資組合報酬率 - 無風險利率) / 投資組合波動率，用來衡量投資組合每單位波動風險，可以獲得多少超過無風險比率的報酬
#簡單來說就是把波動納進來，看看報酬率是否值得承擔風險，越高代表越值得投資。無風險利率就是你把錢放在銀行定存的利率(或者是我自己定的一個利率)，國泰世華定期儲蓄是1.72%
# risk_free_rate = 0.0172 #無風險利率
# sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
# print("Sharpe Ratio:", sharpe_ratio)

#Heatmap of Correlation Matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(result_corr, annot = True, fmt = ".2f", cmap = "coolwarm")
# plt.title("Correlation Heatmap of Stock Returns")
# plt.savefig("images/correlation_heatmap.png", dpi=300, bbox_inches="tight")
# plt.show()

#Protfolio comparison 各比率比較，上面的是單一比率，下面是多個比率比較
mean_daily_returns = returns[["2330.TW", "0050.TW"]].mean() #每日平均報酬率
print("Mean Daily Returns:", mean_daily_returns)
portfolio_daily_return = np.sum(mean_daily_returns.to_numpy() * portfolio_weights, axis = 1) #投資組合每日平均報酬率，axis = 1 代表橫向相加，axis = 0 代表縱向相加，不加會全部加一起，只會有一個數
portfolio_annualized_return = portfolio_daily_return * 252 #年化報酬率
print("Annualized Return:", portfolio_annualized_return)

portfolio_variance = []
for weights in portfolio_weights:
    variance = weights.T @ cov_matrix @ weights
    portfolio_variance.append(variance)

print("Portfolio Variance:", portfolio_variance)

portfolio_annual_volatility = np.sqrt(portfolio_variance) * np.sqrt(252) #年化波動率
print("Portfolio Annualized Volatility:", portfolio_annual_volatility)

sharpe_ratio_all = (portfolio_annualized_return - 0.0172) / portfolio_annual_volatility #夏普比率
print("Sharpe Ratio:", sharpe_ratio_all)

#DataFrame for Portfolio Comparison
portfolio_comparison = pd.DataFrame({
    "Weights": portfolio_weights.tolist(),
    "Annualized Return": portfolio_annualized_return,
    "Annualized Volatility": portfolio_annual_volatility,
    "Sharpe Ratio": sharpe_ratio_all
})

print(portfolio_comparison)

# plotting the highest sharpe ratio portfolio
best_index = np.argmax(sharpe_ratio_all) #找出夏普比率最高的投資組合的位置，台積電/0050 最佳為3
print(best_index)
print("Best Portfolio Weights:", portfolio_weights[best_index])
print(sharpe_ratio_all[best_index])

plt.figure(figsize=(8, 6))
plt.scatter(
    portfolio_annual_volatility, 
    portfolio_annualized_return,
    color='blue',
    label='Portfolios'
)

# Risk-Return Scatter Plot
plt.scatter(
    portfolio_annual_volatility[best_index], 
    portfolio_annualized_return[best_index], 
    color='red', 
    marker='*', 
    s=150, 
    label='Highest Sharpe'
)

labels = ["100/0", "75/25", "50/50", "25/75", "0/100"]
for i in range(len(labels)):
    plt.annotate(
        f"{labels[i]}\nSharpe: {sharpe_ratio_all[i]:.3f}",
        (
            portfolio_annual_volatility[i],
            portfolio_annualized_return[i]
        ),
        xytext = (8,5),
        textcoords = "offset points",
    )

plt.gca().xaxis.set_major_formatter(PercentFormatter(1)) # PercentFormatter(1) 是指把數值轉換成百分比格式，1代表100%，所以會把0.1顯示成10%
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title("Portfolio Risk-Return Comparison")
plt.xlabel("Annualized Volatility (Risk)")
plt.ylabel("Annualized Return")
plt.legend()
plt.show()

#Portfolio Optimization
tsmc_weights = np.linspace(0, 1, 101) #台積電權重從0到1，分成100個點
etf0050_weight = 1 - tsmc_weights #0050權重為1減去台積電權重
portfolio_weights_101 = np.column_stack((tsmc_weights, etf0050_weight)) #將台積電權重和0050權重組合成一個二維陣列，第一欄是台積電權重，第二欄是
print(portfolio_weights_101.shape)
print(portfolio_weights_101[:5]) #印出前五筆資料

portfolio_daily_return_101 = portfolio_weights_101 @ mean_daily_returns.to_numpy() #投資組合每日平均報酬率，使用矩陣乘法
portfolio_annualized_return_101 = portfolio_daily_return_101 * 252 #年化報酬率

#Portfolio Variances
portfolio_variance_101 = []

for weights in portfolio_weights_101:
    variance = weights.T @ cov_matrix.to_numpy() @ weights
    portfolio_variance_101.append(variance)

# 101 組年化波動率
portfolio_annual_volatility_101 = np.sqrt(portfolio_variance_101) * np.sqrt(252) #年化波動率
print("Portfolio Annualized Volatility 101:", portfolio_annual_volatility_101[:5])

#Sharpe Ratio 101
sharpe_ratio_101 = (portfolio_annualized_return_101 - 0.0172) / portfolio_annual_volatility_101 #夏普比率
print("Sharpe Ratio 101:", sharpe_ratio_101[:5])

# Best sharpe ratio portfolio
best_index_101 = np.argmax(sharpe_ratio_101)
print("Best Portfolio Weights 101:", portfolio_weights_101[best_index_101])

print("Best Index:", best_index_101)
print("Best Weights:", portfolio_weights_101[best_index_101])
print("Highest Sharpe:", sharpe_ratio_101[best_index_101])
print("Annual Return:", portfolio_annualized_return_101[best_index_101])
print("Annual Volatility:", portfolio_annual_volatility_101[best_index_101])

#Plotting the Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(
    portfolio_annual_volatility_101,
    portfolio_annualized_return_101,
    color = 'blue',
    alpha = 0.5,
    label = 'Portfolios'
)

# Highlighting the best portfolio
plt.scatter(
    portfolio_annual_volatility_101[best_index_101],
    portfolio_annualized_return_101[best_index_101],
    color = 'red',
    marker = '*',
    s = 200,
    label = 'Highest Sharpe Ratio'
)

plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title("Portfolio Risk–Return Curve (101 Allocations)")
plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")
plt.legend()
plt.show()