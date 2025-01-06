# Research objective 

We aim at modelling and predicting performance in Team Sport 

Provided with the position of Rugby Union player, we want to predict the evolution of a "success" variable. The selected "success" variable is the position of the ball in the field.

# Problem formulation

Let $X_{(i,t)}$ be the position vector of player $i \in \{1, ..., 15\}$ at time $t$. $X_{(i,t)}$ is 2-dimensional.  
Let $X_t$ be $\{X_{(1,t)}, ..., X_{(15,t)}\}$.  
Let $s$ be the sequence length and $p$ be the prediction length.  
For readability purpose, for any $t$ we define $(X_S)_t = \{X_{(t)}, X_{(t+1)}, ... X_{(t+s)}\}$   
Let $A_t$ be the value of the "success" variable at time $t$, and $(A_\tau)_t = A_{(t + s + p)} - A_{(t + s)}$ be the difference between the "success" variable at the end of the prediction horizon and the "success" variable at the end of the sequence.  

Our objective can be formulated as : Predicting $A_\tau$ given $X_S$


# Pseudo related work

We distinguished two communities working of problem whose formulation is rather similar as ours :
* **Time Series** : 
  * Similarities :
    * Predicting future based on past
    * Task can be either multivariate predicts multivariate, multivariate predicts univariate, univariate predicts univariate. 
  * Differences :
    * One single and uninterrupted Time Series
    * Does not necessarily involve position data (benchmark datasets do not)
  * Reference datasets : 8 datasets, used in all papers I read
  * Activity : +++
  * Code availability : +++
  * Metrics : MSE, MAE for LTF, SMAPE and MASE for STF
* **Multi-agent trajectory prediction** : 
  * Similarities :
    * Predicting future based on past
    * Position data
    * Use of a scene (in our case, the limits of the field)
  * Differences :
    * Outputting various trajectories (which is not an issue)
  * Reference datasets : 3/4, not systematically used in all papers
  * Activity : +
  * Code availability : ~
  * Metrics : Miss Rate, ADE, FDE 

# Implementation
