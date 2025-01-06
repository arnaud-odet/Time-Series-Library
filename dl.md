# Research objective 

We aim at modelling and predicting performance in Team Sport 

Provided with the position of Rugby Union player, we want to predict the evolution of a "success" variable. The selected "success" variable is the position of the ball in the field.

# Problem formulation

Let $X_{(i,t)}$ be the position vector of player $i \in \{1, ..., 15\}$ at time $t$. $X_{(i,t)}$ is 2-dimensional.  
Let $X_t$ be $\{X_{(1,t)}, ..., X_{(15,t)}\}$.  
Let $s$ be the sequence length and $p$ be the prediction length.  
For readability purpose, for any $t$ we define $(X_S)_t = \{X_{(t)}, X_{(t+1)}, ... X_{(t+s)}\}$   
Let $A_t$ be the value of the "success" variable at time $t$, and $\tau = t + s + p$ be the timestamp at the end of the prediction period.  

Or objective can be formulated as 




# Implementation
