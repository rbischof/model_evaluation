# Model evaluation
plot_functions.py contains a set of functions that can be called to evaluate the performance of a model, as well as comparing multiple models in a taylor diagram.

`accuracy_diagonal_plot(x, y, predictions, labels, path)` takes two-dimensional np.array x (# of samples, # of features), two-dimensional np.array y (# of samples, # of output variables), two-dimensional np.array predictions (# of samples, # of output variables), list of output variable names of length (# of output variables), path as string defining the location where to save the figures to.

![Diagonal Accuracy Plot](/examples/diagonal_match.png)

`mean_correction_factor_plot(y, predictions, labels, path)` takes two-dimensional np.array y (# of samples, # of output variables), two-dimensional np.array predictions (# of samples, # of output variables), list of output variable names of length (# of output variables), path as string defining the location where to save the figures to.

![Mean Correction Factor Plot](/examples/mean_correction_factor.png)

`qq_plot(y, predictions, labels, path)` takes two-dimensional np.array y (# of samples, # of output variables), two-dimensional np.array predictions (# of samples, # of output variables), list of output variable names of length (# of output variables), path as string defining the location where to save the figures to.

![QQ Plot](/examples/qq_plot.png)

`ratio_plot(y, predictions, labels, path)` takes two-dimensional np.array y (# of samples, # of output variables), two-dimensional np.array predictions (# of samples, # of output variables), list of output variable names of length (# of output variables), path as string defining the location where to save the figures to.

![Ratio Plot](/examples/_true_prediction.png)

`taylor_plot(model_names, y, predictions, labels, path)` takes list of model names, two-dimensional np.array y (# of samples, # of output variables), list of two-dimensional np.array predictions (# of samples, # of output variables) of same length as model_names, list of output variable names of length (# of output variables), path as string defining the location where to save the figures to.

![Taylor Plot](/examples/taylor_diagram.png)

