import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
import pmdarima as pm
from sklearn.model_selection import train_test_split
from datetime import datetime
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
import pickle


"""Runtime configuration of Matplotlib"""
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200  # Image pixel density
plt.rcParams['figure.dpi'] = 200  # Resolution for on-screen display

"""Function to check the stationarity of the data, to determinne the value of d in ARIMA(p,d,q)."""
def test_stationarity(data):
    data_diff = data.diff()
    plt.plot(data, label = 'Number of Internet Connections', color = 'black')
    plt.plot(data_diff, label = '1st Order Differencing', color = 'red')
    plt.legend()   
    plt.savefig(r'Pictures/stationarity.png')
    plt.show()

"""Function to calculate Autocorrelation and Partial Autocorrelation"""
def ACF_PACF(data):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags=12,ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize = 20 )
    plt.yticks(fontsize = 20 )
    fig.tight_layout()
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=12, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    plt.xticks(fontsize = 20 )
    plt.yticks(fontsize = 20 )
    plt.savefig(r'Pictures/ACF_PACF.png')
    plt.show()

"""Function to draw a scatter plot of the data to check for seasonality and determine value of p and q in ARIMA(p,d,q)."""
def scatter_plot(data):
    lags=9
    ncols=3
    nrows=int(np.ceil(lags/ncols))

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows))

    for ax, lag in zip(axes.flat, np.arange(1,lags+1, 1)):
        lag_str = 't-{}'.format(lag)
        X = (pd.concat([data, data.shift(-lag)], axis=1,
                    keys=['y'] + [lag_str]).dropna())

        X.plot(ax=ax, kind='scatter', y='y', x=lag_str)
        corr = X.corr().values[0][1]
        ax.set_ylabel('Original')
        ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr))
        ax.set_aspect('equal')
        sns.despine()

    fig.tight_layout()
    plt.savefig(r'Pictures/scatterplot.png')
    plt.show()

def decide_PQ(y_Train,x_Train):   
    arima401 = sm.tsa.SARIMAX(endog=y_Train,exog = x_Train, order=(4,0,1))
    model_results = arima401.fit()

    p_min = 0
    d_min = 0
    q_min = 0
    p_max = 4
    d_max = 0
    q_max = 4

    results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

    for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
        if p==0 and d==0 and q==0:
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            continue
    
        try:
            model = sm.tsa.SARIMAX(endog=y_Train,exog=x_Train, order=(p, d, q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                              )
            results = model.fit()
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
        except:
            continue
    results_bic = results_bic[results_bic.columns].astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f', 
                 annot_kws = {'size':20}
                 );
    #ax.set_title('BIC');
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(r'Pictures/BIC.png')
    plt.show()
def pred(y_train, x_train):
    arima001 = sm.tsa.SARIMAX(y_train, x_train, order=(1,0,2))
    model_fit = arima001.fit()
    
    print(model_fit.summary())
    print(model_fit.params)
    print(model_fit.aic)
    return model_fit

def plot_actual_vs_predicted(actual, predicted, title='Actual vs Predicted', xlabel='index', ylabel='Data'):

    plt.figure(figsize=(15, 7))
    plt.plot(actual, label='Actual', color='blue', alpha=0.6)  # Plot the actual values
    plt.plot(predicted, label='Predicted', color='red', linestyle='--', alpha=0.7)  # Plot the predicted values
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()  # Show the legend
    plt.grid(True)  # Show grid
    plt.tight_layout()  # Fit the plot neatly
    plt.savefig(r'Pictures/actualvspredicted.png')
    plt.show()
def mse(actual, predicted):
    return mean_squared_error(actual, predicted)

if __name__ == '__main__':
    """Open the pickle file 'grouped_datetime.pickle' and store its content in a DataFrame"""
    with open('/Users/shrey_98/Project_CS_UserVsSpecific/grouped_datetime.pkl', 'rb') as file:
        df = pickle.load(file)
    """
    test_stationarity(df['internet'])
    ACF_PACF(df['internet'])
    scatter_plot(df['internet'])
    """
    """Dropping the columns that are not required and implementing StandardScaler on the data"""
    df.drop(['sms_in','sms_out','call_in','call_out'],axis=1,inplace=True)
    scaler = StandardScaler()
    df['internet'] = scaler.fit_transform(df[['internet']])
    """Split the data into x_train,x_test,y_train,y_test"""
    x_train,x_test,y_train,y_test = train_test_split(df[['internet']],df[['internet']],test_size=0.2,random_state=0,shuffle=False)
    decide_PQ(x_train,y_train)
    model_fit = pred(y_train, x_train)
    predictions = model_fit.predict(start=y_train.index[0], end=y_train.index[-1])
    predictions_2d = predictions.to_numpy().reshape(-1, 1) 
    y_train_2d = y_train.to_numpy().reshape(-1, 1)  
    """Now perform the inverse transform"""
    predictions = scaler.inverse_transform(predictions_2d)
    y_train = scaler.inverse_transform(y_train_2d)
    plot_actual_vs_predicted(y_train[:,0], predictions[:,0])
   # print(r2_score(y_train[:,0],predictions[:,0]))
    print(sqrt(mean_squared_error(y_train[:, 0], predictions[:, 0])))
    

    


