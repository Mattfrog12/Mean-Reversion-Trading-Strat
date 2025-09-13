**Mean Reversion Trading Strategy**

This strategy takes stock ticker data from yfinance and uses random forests to build a 
trading model with buy and sell signals that outperforms the buy and hold benchmark by
18 percent. Since this model is tailored for mean reverting stocks (more on that later), we also
must select our stocks carefully. With some tweaks, this is supposed to follow the methodology 
presented in "Predicting the Direction of Stock Market Prices using Random Forest" (Khaidem, Saha, Dey). 

There are a few important steps, so let's go through them slowly. 

Step 1: Data Gathering

This is where we use yfinance to pull data about a specific ticker. In our case, 
closing prices were the most important. 

Step 2: Data Smoothing 

We care more about recent data and we want our model to do the same, so we use 
exponential smoothing so that the model will be swayed more by recent data.
All this smoothing does is apply a factor of less than 1 to each data point, and 
multiplies through by that factor at each time step backwards so that older data
is less consequential. 

Step 3: Feature Creation

Now that we have our data, we want to create features using that data. 
The paper recommends Relative Strength Index, Stochastic Oscillator, 
Moving Average Convergence Divergence, Price Rate of Change, and 
On Balance Volume. I added volatility, momentum, and lag-1 autocorrelation. 

Important: Be sure to shift your features and target array properly, so that 
the model does not use today's data-or even tomorrow's data-to predict today's prices. 

Step 4: Define Relevant Features and Targets

We have buy and sell vectors with the same features (not necessary) and defined
target arrays for buying and selling. Note that these are seperate, and that we are 
really using two models.

Step 5: Add Model Optimizers like grid_search_cv, Define Train versus Test

This step isn't necessary and may actually lead to overfitting if too many parameters
are inputted, but optimizing for every stock is expensive so this is a tradeoff. Lastly, 
we need some data to be training data and some to be test data, so split at a reasonable 
point. We want our model to predict more recent moves.

Step 6: Train and Test Model

Finally, we can begin the model training and testing. We loop through the data 50 times
as to reduce variance in our results, because remember that there is a random element
in random forests. 

Step 7: Model Evaluation

There are many different metrics to look at, but we chose OOB score, accuracy, MDI score,
and simply model returns to gauge how well the model performs. 
