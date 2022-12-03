# Knowledge Distillation

## Background
When training machine learning models, the ensamble method is very popular but it comes the draw back that it is computationally expensive. Other kinds of models
like ones that use a single model are also very computationally expensive. The training and predicting of these kinds of models may be okay on powerful computers
but when deploying these models to consumers who have signifcantly weaker machines the models essentially become useless as they take too long to make a prediction.
This is where compression comes in, compression means to take a trained model and compression so it can run a smaller or weaker computer.

# Knowledge Distillation
You can take a trained model and extract the knowledge from the vary large trained model and put it in a much smaller model making it available to be used on alot more machines.
The object of machine learning models is to generalize new data but this cant be done if the training model doesnt have the data needed to generalize, but using knowledge
distillation we can train the smaller model to generalize data in the same way that the larger model would.

Neural Networks use the softmax function to create probabilities of each class, using this we can transfer the knowledge to the smaller model. The smaller model is
trained using a transfer set and the output of the softmax function, which has the probabilities of each class. 
