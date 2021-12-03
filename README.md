# Model evaluation
plot_functions.py contains a set of functions that can be called to evaluate the performance of a model, as well as comparing multiple models in a taylor diagram.

`accuracy_diagonal_plot(x, y, predictions, labels, path)` takes two-dimensional np.array x (# of samples, # of features), two-dimensional np.array y (# of samples, # of output variables), two-dimensional np.array predictions (# of samples, # of output variables), list of output variable names of length (# of output variables), path as string defining the location where to save the figures to.

![Diagonal Accuracy Plot](/examples/diagonal_match.png)

