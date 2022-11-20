# BOHB
BOHB is the method of combining Bayesian Optimization and Hyperband optimization to get the best parts of each optimization method. BOHB has stronger performance all of the time as well as fast convergence on the optimal configuration. The figure below shows how HOHB compares to other methods of optimization. BOHB is signifcantly faster.

<img width="519" alt="Screen Shot 2022-11-20 at 8 05 46 AM" src="https://user-images.githubusercontent.com/59149625/202903502-a2d17b73-b323-4e98-bc9c-38fecd538082.png">

To understand BOHB first we should look at its components, Bayesian Optimization and Hyperband optimization.

## Hyperband Optimization
Hyperband is a bandit strategy that dynamically allocates resources to some random configureation and then halves them in succession. The halving allows the algorithm to end configurations that have poor performance. The downside of Hyperband optimization is that the configurations are random and the algorithms doesnt learn from its previous iterations.

## Bayesian Optimization
Bayesian optimization is used to find the optimal function when the function is hard to calculate or is too expensive to calculate. Bayesian optimziation is based on Bayes Theorm and works by building a probabilistic model of the objective function. The probabilistic model is called the surogate function and this function is searched through with another function to find the optimal parameters of an already well performing model to tune its parameters.

# BOBH Algorithm
BOHB works by using Hyperband optimation to determine how many configurations should be evaluated. The configurations, instead of being random like in classic Hyperband optimization, are found via model-based search. Once the configurations are selected the algorithms continues following the path of Hyperband optimzation and successively halves the resources. The performance of all the configurations are then tracked to be used later.

The algorithms then continues the path of Hyperband optimization and selects some budgets. The selection process for the next configuration is adapted from Bayesian Optimization (Algorithms Pseudocode in figure 1). Eventually through the optimization process more configurations are created with bigger and bigger budgets, but this is offset by BOHB using the model with the largest budget which has the most observations.

<img width="510" alt="Screen Shot 2022-11-20 at 8 04 27 AM" src="https://user-images.githubusercontent.com/59149625/202903423-489314b5-3b5f-44ec-ba20-5d145417f6d7.png">

While BOHB has its own hyperparamters, these hyperparameters have little effect.

