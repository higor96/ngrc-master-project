from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from typing import List, Tuple

# Update default matplotlib parameters for consistent styling
plt.rcParams.update({
    'font.size': 10,
    'lines.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
})

def plot_comparison(
    y_true_train: np.ndarray, 
    y_pred_train: np.ndarray, 
    t_eval_train: np.ndarray,
    y_true_test: np.ndarray, 
    y_pred_test: np.ndarray, 
    t_eval_test: np.ndarray, 
    y_labels: List[str]
) -> plt.Figure:
    """
    Create a comparison plot for both training and testing phases.

    This function plots side-by-side comparisons for numerical simulation and NG-RC predictions
    for both training and testing phases.

    Parameters:
        y_true_train (np.ndarray): True values during training.
        y_pred_train (np.ndarray): Predicted values during training.
        t_eval_train (np.ndarray): Time evaluation points for training.
        y_true_test (np.ndarray): True values during testing.
        y_pred_test (np.ndarray): Predicted values during testing.
        t_eval_test (np.ndarray): Time evaluation points for testing.
        y_labels (List[str]): List of labels for each state variable.

    Returns:
        plt.Figure: The matplotlib figure object with the plots.
    """
    n_labels = len(y_labels)
    fig, axs = plt.subplots(n_labels, 2, figsize=(10, 10), layout='constrained')

    # Plot training data in the first column
    for i, ax_row in enumerate(axs):
        ax_row[0].plot(t_eval_train, y_true_train[i, :], color='b', label='Numerical Simulation')
        ax_row[0].plot(t_eval_train, y_pred_train[i, :], color='r', label='NG-RC Prediction')
        ax_row[0].set_ylabel(y_labels[i])
        # Share x-axis with the first subplot in the first column
        ax_row[0].sharex(axs[0][0])

    # Plot testing data in the second column
    for i, ax_row in enumerate(axs):
        ax_row[1].plot(t_eval_test, y_true_test[i, :], color='b', label='Numerical Simulation')
        ax_row[1].plot(t_eval_test, y_pred_test[i, :], color='r', label='NG-RC Prediction')
        ax_row[1].set_ylabel(y_labels[i])
        ax_row[1].sharex(axs[0][1])
        ax_row[1].sharey(axs[i][0])

    # Set titles and labels for the first row and last row subplots
    axs[0][0].set_title('Training Phase')
    axs[0][1].set_title('Testing Phase')
    axs[-1][0].set_xlabel('Time [s]')
    axs[-1][1].set_xlabel('Time [s]')

    # Only display outer labels
    for ax in fig.get_axes():
        ax.label_outer()

    # Create a common legend for the figure
    handles, labels = axs[0][0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), 
        by_label.keys(), 
        loc='lower center',      
        bbox_to_anchor=(0.5, -0.05), 
        ncol=2
    )
    
    fig.align_labels()
    
    return fig

def plot_comparison_only_testing(
    y_true_test: np.ndarray, 
    y_pred_test: np.ndarray, 
    t_eval_test: np.ndarray, 
    y_labels: List[str]
) -> plt.Figure:
    """
    Create a comparison plot for the testing phase only.

    This function plots a vertical stack of subplots comparing numerical simulation
    and NG-RC predictions during the testing phase.

    Parameters:
        y_true_test (np.ndarray): True values during testing.
        y_pred_test (np.ndarray): Predicted values during testing.
        t_eval_test (np.ndarray): Time evaluation points for testing.
        y_labels (List[str]): List of labels for each state variable.

    Returns:
        plt.Figure: The matplotlib figure object with the plots.
    """
    n_labels = len(y_labels)
    fig, axs = plt.subplots(n_labels, 1, figsize=(10, 10), layout='constrained')

    # If there is only one label, ensure axs is iterable
    if n_labels == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        ax.plot(t_eval_test, y_true_test[i, :], color='b', label='Numerical Simulation')
        ax.plot(t_eval_test, y_pred_test[i, :], color='r', label='NG-RC Prediction')
        ax.set_ylabel(y_labels[i])
        ax.sharex(axs[0])
    
    axs[-1].set_xlabel('Time [s]')

    for ax in fig.get_axes():
        ax.label_outer()

    # Create a common legend for the figure
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), 
        by_label.keys(), 
        loc='lower center',      
        bbox_to_anchor=(0.5, -0.05), 
        ncol=2
    )
    
    fig.align_labels()
    
    return fig

def plot_scatter_comparison(
    y_true_train: np.ndarray, 
    y_pred_train: np.ndarray, 
    y_true_test: np.ndarray, 
    y_pred_test: np.ndarray, 
    var_indices: List[int], 
    axis_labels: List[str],
    train_title: str = "Training Phase", 
    test_title: str = "Testing Phase"
) -> plt.Figure:
    """
    Creates scatter plots comparing numerical simulation and NG-RC predictions for
    specified variables, for both training and testing phases.
    
    Parameters:
        y_true_train (np.ndarray): True training data (variables x time points).
        y_pred_train (np.ndarray): Predicted training data.
        y_true_test (np.ndarray): True testing data.
        y_pred_test (np.ndarray): Predicted testing data.
        var_indices (List[int]): List of indices of the variables to plot.
        axis_labels (List[str]): List of axis labels (e.g., "θ₁ [rad]").
        train_title (str): Title for the training-phase subplots.
        test_title (str): Title for the testing-phase subplots.
        
    Returns:
        plt.Figure: The generated figure with subplots.
    """
    n_vars = len(var_indices)
    fig, axs = plt.subplots(n_vars, 2, figsize=(10, 5 * n_vars), layout='constrained')
    # Ensure axs is always a list of rows
    if n_vars == 1:
        axs = [axs]

    for i, idx in enumerate(var_indices):
        # Training scatter plot
        ax_train = axs[i][0]
        ax_train.scatter(y_true_train[idx, :], y_pred_train[idx, :], color='purple', s=3)
        min_val = min(np.min(y_true_train[idx, :]), np.min(y_pred_train[idx, :]))
        max_val = max(np.max(y_true_train[idx, :]), np.max(y_pred_train[idx, :]))
        ax_train.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
        ax_train.set_xlabel(f"{axis_labels[i]} (Numerical Simulation)")
        ax_train.set_ylabel(f"{axis_labels[i]} (NG-RC Prediction)")
        if i == 0:
            ax_train.set_title(train_title)
        ax_train.legend()

        # Testing scatter plot
        ax_test = axs[i][1]
        ax_test.scatter(y_true_test[idx, :], y_pred_test[idx, :], color='purple', s=3)
        min_val = min(np.min(y_true_test[idx, :]), np.min(y_pred_test[idx, :]))
        max_val = max(np.max(y_true_test[idx, :]), np.max(y_pred_test[idx, :]))
        ax_test.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
        ax_test.set_xlabel(f"{axis_labels[i]} (Numerical Simulation)")
        ax_test.set_ylabel(f"{axis_labels[i]} (NG-RC Prediction)")
        if i == 0:
            ax_test.set_title(test_title)
        ax_test.legend()

    return fig


def plot_phase_comparison(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    phase_vars: List[Tuple[int, int]],
    axis_labels: List[Tuple[str, str]],
    train_title: str = "Training Phase",
    test_title: str = "Testing Phase"
) -> plt.Figure:
    """
    Creates phase plots comparing numerical simulation and NG-RC predictions for
    specified variable pairs (e.g., position vs. velocity), for both training and testing phases.
    
    Parameters:
        y_true_train (np.ndarray): True training data (variables x time points).
        y_pred_train (np.ndarray): Predicted training data.
        y_true_test (np.ndarray): True testing data.
        y_pred_test (np.ndarray): Predicted testing data.
        phase_vars (List[Tuple[int, int]]): List of tuples where each tuple contains the indices 
            for the x and y variables (e.g., (0, 1) for φ and ω).
        axis_labels (List[Tuple[str, str]]): List of tuples with axis labels (e.g., ("θ₁ [rad]", "ω₁ [rad/s]")).
        train_title (str): Title for the training-phase subplots.
        test_title (str): Title for the testing-phase subplots.
        
    Returns:
        plt.Figure: The generated figure with subplots.
    """
    n_pairs = len(phase_vars)
    fig, axs = plt.subplots(n_pairs, 2, figsize=(10, 5 * n_pairs), layout='constrained')
    if n_pairs == 1:
        axs = [axs]

    for i, (x_idx, y_idx) in enumerate(phase_vars):
        # Training phase plot
        ax_train = axs[i][0]
        ax_train.plot(y_true_train[x_idx, :], y_true_train[y_idx, :], 'b-', label="Numerical Simulation")
        ax_train.plot(y_pred_train[x_idx, :], y_pred_train[y_idx, :], 'r--', label="NG-RC Prediction")
        ax_train.set_xlabel(axis_labels[i][0])
        ax_train.set_ylabel(axis_labels[i][1])
        if i == 0:
            ax_train.set_title(train_title)
        ax_train.legend()

        # Testing phase plot
        ax_test = axs[i][1]
        ax_test.plot(y_true_test[x_idx, :], y_true_test[y_idx, :], 'b-', label="Numerical Simulation")
        ax_test.plot(y_pred_test[x_idx, :], y_pred_test[y_idx, :], 'r--', label="NG-RC Prediction")
        ax_test.set_xlabel(axis_labels[i][0])
        ax_test.set_ylabel(axis_labels[i][1])
        if i == 0:
            ax_test.set_title(test_title)
        ax_test.legend()

    return fig


def plot_scatter_comparison_only_testing(
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    var_indices: List[int],
    axis_labels: List[str]
) -> plt.Figure:
    """
    Creates scatter plots comparing numerical simulation and NG-RC predictions for
    specified variables during the testing phase only.
    
    Parameters:
        y_true_test (np.ndarray): True testing data (variables x time points).
        y_pred_test (np.ndarray): Predicted testing data.
        var_indices (List[int]): List of indices of the variables to plot.
        axis_labels (List[str]): List of axis labels (e.g., "θ₁ [rad]").
        
    Returns:
        plt.Figure: The generated figure with subplots.
    """
    n_vars = len(var_indices)
    fig, axs = plt.subplots(n_vars, 1, figsize=(10, 5 * n_vars), layout='constrained')
    if n_vars == 1:
        axs = [axs]

    for i, idx in enumerate(var_indices):
        ax = axs[i]
        ax.scatter(y_true_test[idx, :], y_pred_test[idx, :], color='purple', s=3)
        min_val = min(np.min(y_true_test[idx, :]), np.min(y_pred_test[idx, :]))
        max_val = max(np.max(y_true_test[idx, :]), np.max(y_pred_test[idx, :]))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
        ax.set_xlabel(f"{axis_labels[i]} (Numerical Simulation)")
        ax.set_ylabel(f"{axis_labels[i]} (NG-RC Prediction)")
        ax.legend()

    return fig


def plot_phase_comparison_only_testing(
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    phase_vars: List[Tuple[int, int]],
    axis_labels: List[Tuple[str, str]]
) -> plt.Figure:
    """
    Creates phase plots comparing numerical simulation and NG-RC predictions for
    specified variable pairs during the testing phase only.
    
    Parameters:
        y_true_test (np.ndarray): True testing data (variables x time points).
        y_pred_test (np.ndarray): Predicted testing data.
        phase_vars (List[Tuple[int, int]]): List of tuples with indices for x and y variables (e.g., (0, 1)).
        axis_labels (List[Tuple[str, str]]): List of tuples with axis labels (e.g., ("θ₁ [rad]", "ω₁ [rad/s]")).
        
    Returns:
        plt.Figure: The generated figure with subplots.
    """
    n_pairs = len(phase_vars)
    fig, axs = plt.subplots(n_pairs, 1, figsize=(10, 5 * n_pairs), layout='constrained')
    if n_pairs == 1:
        axs = [axs]

    for i, (x_idx, y_idx) in enumerate(phase_vars):
        ax = axs[i]
        ax.plot(y_true_test[x_idx, :], y_true_test[y_idx, :], 'b-', label="Numerical Simulation")
        ax.plot(y_pred_test[x_idx, :], y_pred_test[y_idx, :], 'r--', label="NG-RC Prediction")
        ax.set_xlabel(axis_labels[i][0])
        ax.set_ylabel(axis_labels[i][1])
        ax.legend()

    return fig

def get_feature_labels(
    input_dimension: int,
    delay_taps: int,
    poly_combinations: List[Tuple[int, ...]],
    include_constant: bool,
    param_names: List[str],
    var_names: List[str]
) -> List[str]:
    """
    Generate a list of feature labels for the NG-RC model.
    
    This function creates labels for:
      1. (Optionally) A constant term.
      2. Linear features using delay taps.
      3. Polynomial (nonlinear) features using provided index combinations.
      4. Extra parameters.
    
    Parameters:
        input_dimension (int): The number of input variables.
        delay_taps (int): The number of delay taps.
        poly_combinations (List[Tuple[int, ...]]): List of tuples representing the index combinations for polynomial features.
        include_constant (bool): Whether to include a constant feature.
        param_names (List[str]): Names of extra parameters.
        var_names (List[str]): Names of the input variables (e.g., ["\\theta_1", "\\dot\\theta_1", ...]).
    
    Returns:
        List[str]: A list of feature labels in LaTeX formatted strings.
    """
    labels: List[str] = []
    
    # 1. Constant feature (if included)
    if include_constant:
        labels.append("const")
        labels.append("$\\text{const}$")
    
    # 2. Linear features (delay taps)
    for d in range(delay_taps):
        for i in range(input_dimension):
            if d == 0:
                labels.append(f"$ {var_names[i]}(t) $")
            else:
                labels.append(f"$ {var_names[i]}(t-{d}) $")
                
    # 3. Polynomial (nonlinear) features
    for comb in poly_combinations:
        # Generate the label for each index in the combination (without $ symbols)
        labels_list: List[str] = []
        for idx in comb:
            delay = idx // input_dimension
            var = idx % input_dimension
            if delay == 0:
                var_label = f"{var_names[var]}(t)"
            else:
                var_label = f"{var_names[var]}(t-{delay})"
            labels_list.append(var_label)
        
        # Group the same labels using a counter
        freq = Counter(labels_list)
        parts: List[str] = []
        # Sort keys to maintain consistency in order
        for key in sorted(freq.keys()):
            count = freq[key]
            if count == 1:
                parts.append(key)
            else:
                parts.append(f"{key}^{{{count}}}")
        # Join parts with a multiplication dot
        poly_label = " \\cdot ".join(parts)
        labels.append(f"$ {poly_label} $")
        
    # 4. Extra parameter labels
    for param in param_names:
        labels.append(f"$ {param} $")
    
    return labels

def plot_W_out(
    W_out: np.ndarray,
    model_params: any,  # Expected to have attributes: polynomial_order, input_dimension, delay_taps, include_constant
    feature_vec_obj: any,  # Expected to have methods: linear_feature_vector_size() and polynomial_combinations()
    param_names: List[str],
    var_names: List[str]
) -> plt.Figure:
    """
    Plot a horizontal bar chart of the output weights (W_out) assigned to each feature.

    The function generates feature labels by calling get_feature_labels and then reverses
    the order of both weights and labels for better visualization. It creates one subplot per output variable.

    Parameters:
        W_out (np.ndarray): Output weight matrix of shape (num_outputs, num_features).
        model_params: Object containing model parameters with attributes such as input_dimension, delay_taps, and include_constant.
        feature_vec_obj: Object with methods to compute linear feature size and polynomial combinations.
        param_names (List[str]): Names of extra parameters to be included in the labels.
        var_names (List[str]): List of variable names for the outputs (e.g., ["$x$", "$y$", "$z$"]).
        
    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    # Extract parameters from the model parameters object
    input_dimension: int = model_params.input_dimension
    delay_taps: int = model_params.delay_taps
    include_constant: bool = model_params.include_constant

    # Get the size of the linear feature vector and polynomial combinations from the feature vector object
    d_lin: int = feature_vec_obj.linear_feature_vector_size()
    poly_combinations: List[Tuple[int, ...]] = feature_vec_obj.polynomial_combinations()

    num_outputs, num_features = W_out.shape
    # Generate labels for features using the provided var_names
    labels: List[str] = get_feature_labels(input_dimension, delay_taps, poly_combinations, include_constant, param_names, var_names)
    
    # Reverse the order of weights and labels for visualization purposes
    W_out_reversed: np.ndarray = W_out[:, ::-1]
    labels_reversed: List[str] = labels[::-1]
    
    # Check if the number of labels matches the number of features
    if len(labels) != num_features:
        print("Warning: Number of labels does not match the number of features!")
    
    # Create a subplot for each output variable
    fig, axs = plt.subplots(
        1, num_outputs, 
        figsize=(max(12, num_features * 0.2), 3 * num_outputs), 
        constrained_layout=True, 
        sharey=True
    )
    
    # If there is only one output, ensure axs is iterable
    if num_outputs == 1:
        axs = [axs]
        
    for i in range(num_outputs):
        axs[i].barh(range(num_features), W_out_reversed[i, :], color='purple')
        axs[i].set_yticks(range(num_features))
        axs[i].set_yticklabels(labels_reversed)
        axs[i].set_xlabel(f"$W_{{out}}$ - ${var_names[i]}$")
        axs[i].set_xlim(-0.005, 0.005)
        
    plt.suptitle("Weights Assigned to Each Feature")
    plt.show()

    return fig