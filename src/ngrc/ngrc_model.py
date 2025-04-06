import math
import itertools
import numpy as np
from sklearn.metrics import r2_score
from typing import List, Tuple, Dict, Any

class ModelParameters:
    """
    Class to store the parameters for the NG-RC model.
    """
    def __init__(self, polynomial_order: int, input_dimension: int, delay_taps: int, include_constant: bool, ridge_regression: float) -> None:
        """
        Initialize model parameters.
        
        :param polynomial_order: Order of the polynomial expansion.
        :param input_dimension: Dimension of the input data.
        :param delay_taps: Number of delay taps for feature construction.
        :param include_constant: Flag to include a constant term.
        :param ridge_regression: Regularization parameter for ridge regression.
        """
        self.polynomial_order = polynomial_order
        self.input_dimension = input_dimension
        self.delay_taps = delay_taps
        self.include_constant = include_constant
        self.ridge_regression = ridge_regression

    def constant(self) -> int:
        """
        Returns 1 if a constant term is to be included, otherwise 0.
        
        :return: 1 if constant is included, 0 otherwise.
        """
        return 1 if self.include_constant else 0


class DatasetDiscretization:
    """
    Class to handle dataset discretization and splitting into training and testing parts.
    """
    def __init__(self, model_parameters: ModelParameters, dataset_time_array: np.ndarray, train_percentage: float, test_percentage: float) -> None:
        """
        Initialize dataset discretization.
        
        :param model_parameters: An instance of ModelParameters.
        :param dataset_time_array: Array containing time stamps of the dataset.
        :param train_percentage: Percentage of data for training (must be between 0 and 100).
        :param test_percentage: Percentage of data for testing (must be between 0 and 100).
        :raises ValueError: If the sum of train_percentage and test_percentage is not 100.
        """
        if train_percentage + test_percentage != 100:
            raise ValueError("The sum of training and testing percentages must be 100.")
        self.model_parameters = model_parameters
        self.dataset_time_array = dataset_time_array
        self.train_percentage = train_percentage / 100.0
        self.test_percentage = test_percentage / 100.0

    def warmup_pts(self) -> int:
        """
        Number of warm-up points equal to the number of delay taps.
        
        :return: Number of warm-up points.
        """
        return self.model_parameters.delay_taps

    def train_pts(self) -> int:
        """
        Calculate the number of training points.
        
        :return: Number of training points.
        """
        return round((self.max_pts() - self.warmup_pts()) * self.train_percentage)

    def test_pts(self) -> int:
        """
        Calculate the number of testing points.
        
        :return: Number of testing points.
        """
        return round((self.max_pts() - self.warmup_pts()) * self.test_percentage)

    def warmup_train_pts(self) -> int:
        """
        Total points used for warm-up and training.
        
        :return: Warm-up points plus training points.
        """
        return self.warmup_pts() + self.train_pts()

    def max_pts(self) -> int:
        """
        Get the total number of points in the dataset.
        
        :return: Total number of data points.
        """
        return self.dataset_time_array.size


class FeatureVector:
    """
    Class to build feature vectors (linear and nonlinear) for NG-RC.
    """
    def __init__(self, model_parameters: ModelParameters, dataset: np.ndarray, dataset_discretization: DatasetDiscretization, including_all_pol_combination: bool) -> None:
        """
        Initialize the FeatureVector builder.
        
        :param model_parameters: Instance of ModelParameters.
        :param dataset: The dataset as a 2D numpy array (features x time points).
        :param dataset_discretization: An instance of DatasetDiscretization.
        :param including_all_pol_combination: Flag to indicate if all polynomial combinations should be included.
        """
        self.model_parameters = model_parameters
        self.dataset = dataset
        self.dataset_discretization = dataset_discretization
        self.including_all_pol_combination = including_all_pol_combination

    def linear_feature_vector_size(self) -> int:
        """
        Compute the size of the linear feature vector.
        
        :return: Size of the linear feature vector.
        """
        return int(self.model_parameters.input_dimension * self.model_parameters.delay_taps)

    def nonlinear_feature_vector_size(self) -> int:
        """
        Compute the size of the nonlinear feature vector using polynomial expansion.
        
        :return: Size of the nonlinear feature vector.
        """
        n = self.linear_feature_vector_size()
        order = self.model_parameters.polynomial_order
        if self.including_all_pol_combination:
            if order == 1:
                num = math.factorial(n + order - 1)
                den = math.factorial(n - 1) * math.factorial(order)
                return int(num / den)
            else:
                count = 0
                for k in range(2, order + 1):
                    count += math.comb(n + k - 1, k)
                return count
        else:
            num = math.factorial(n + order - 1)
            den = math.factorial(n - 1) * math.factorial(order)
            return int(num / den)
    
    def full_feature_vector_size(self) -> int:
        """
        Calculate the total size of the full feature vector including constant, linear, and nonlinear parts.
        
        :return: Total size of the full feature vector.
        """
        return self.linear_feature_vector_size() + self.nonlinear_feature_vector_size() + self.model_parameters.constant()

    def build_linear_feature_vector(self) -> np.ndarray:
        """
        Build the linear feature vector using delay taps.
        
        :return: A 2D numpy array representing the linear feature vector.
        """
        d_lin = self.linear_feature_vector_size()
        max_pts = self.dataset_discretization.max_pts()
        delay_taps = self.model_parameters.delay_taps
        input_dim = self.model_parameters.input_dimension
        linear_vector = np.zeros((d_lin, max_pts))
        for delay in range(delay_taps):
            for j in range(delay, max_pts):
                linear_vector[input_dim * delay: input_dim * (delay + 1), j] = self.dataset[:, j - delay]
        return linear_vector

    def polynomial_combinations(self) -> List[Tuple[int, ...]]:
        """
        Generate all polynomial index combinations for nonlinear expansion.
        
        :return: List of tuples representing index combinations.
        """
        order = self.model_parameters.polynomial_order
        d_lin = self.linear_feature_vector_size()
        if self.including_all_pol_combination:
            if order == 1:
                return list(itertools.combinations_with_replacement(range(d_lin), order))
            else:
                comb_all = []
                for i in range(2, order + 1):
                    comb_all.extend(list(itertools.combinations_with_replacement(range(d_lin), i)))
                return comb_all
        else:
            return list(itertools.combinations_with_replacement(range(d_lin), order))

    def calc_polynomial_prod(self, combination: Tuple[int, ...], vector: np.ndarray, idx: int) -> float:
        """
        Calculate the product for a given polynomial combination.
        
        This function can be modified to include a specific nonlinearity (e.g., tanh).
        
        :param combination: Tuple of indices representing the combination.
        :param vector: The feature vector (2D numpy array).
        :param idx: The index at which to calculate the product.
        :return: The product as a float.
        """
        # Example: product of the elements in the given combination
        product = np.prod([vector[comp, idx] for comp in combination], axis=0)
        return product

    def build_full_feature_vector(self, linear_vector: np.ndarray) -> np.ndarray:
        """
        Construct the full feature vector (constant, linear, and nonlinear features) for training.
        
        :param linear_vector: Precomputed linear feature vector.
        :return: A 2D numpy array representing the full feature vector.
        """
        dtot = self.full_feature_vector_size()
        d_lin = self.linear_feature_vector_size()
        train_pts = self.dataset_discretization.train_pts()
        cte = self.model_parameters.constant()
        warmup = self.dataset_discretization.warmup_pts()
        warmup_train = self.dataset_discretization.warmup_train_pts()

        full_vector = np.ones((dtot, train_pts))
        # Insert linear features into the full vector
        full_vector[cte: d_lin + cte, :] = linear_vector[:, warmup - 1: warmup_train - 1]
        index_combinations = self.polynomial_combinations()

        cnt = 0
        for comb in index_combinations:
            prod = self.calc_polynomial_prod(comb, linear_vector, slice(warmup - 1, warmup_train - 1))
            full_vector[d_lin + cte + cnt, :] = prod
            cnt += 1
        return full_vector


def compute_training_prediction(linear_vector: np.ndarray, full_vector: np.ndarray, warmup: int, warmup_train: int, train_pts: int, input_dim: int, ridge_param: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the training prediction using ridge regression.
    
    :param linear_vector: Linear feature vector.
    :param full_vector: Full feature vector.
    :param warmup: Number of warmup points.
    :param warmup_train: Total number of warmup and training points.
    :param train_pts: Number of training points.
    :param input_dim: Dimension of the input.
    :param ridge_param: Ridge regression parameter.
    :return: A tuple containing output weights (W_out) and predicted training data (X_predict).
    """
    diff_solution = linear_vector[0:input_dim, warmup:warmup_train] - linear_vector[0:input_dim, warmup - 1:warmup_train - 1]
    dtot = full_vector.shape[0]
    # Compute output weights using ridge regression formula
    W_out = diff_solution @ full_vector.T @ np.linalg.pinv(full_vector @ full_vector.T + ridge_param * np.eye(dtot))
    X_predict = linear_vector[0:input_dim, warmup - 1:warmup_train - 1] + W_out @ full_vector[:, :train_pts]
    return W_out, X_predict


def compute_testing_prediction(linear_vector: np.ndarray, feature_obj: FeatureVector, W_out: np.ndarray, warmup_train: int, test_pts: int, input_dim: int) -> np.ndarray:
    """
    Compute testing prediction based on the trained model.
    
    :param linear_vector: Linear feature vector.
    :param feature_obj: Instance of FeatureVector.
    :param W_out: Output weights from training.
    :param warmup_train: Total warmup and training points.
    :param test_pts: Number of testing points.
    :param input_dim: Input dimension.
    :return: Predicted testing data as a numpy array.
    """
    d_lin = feature_obj.linear_feature_vector_size()
    x_test = np.zeros((d_lin, test_pts))
    x_test[:, 0] = linear_vector[:, warmup_train - 1]
    for j in range(test_pts - 1):
        out_test = np.zeros(feature_obj.full_feature_vector_size())
        cte = feature_obj.model_parameters.constant()
        out_test[cte: d_lin + cte] = x_test[:, j]
        index_combinations = feature_obj.polynomial_combinations()
        cnt = 0
        for comb in index_combinations:
            prod = feature_obj.calc_polynomial_prod(comb, x_test, j)
            out_test[cte + d_lin + cnt] = prod
            cnt += 1
        # Shift previous linear features and update with prediction correction
        x_test[input_dim:d_lin, j + 1] = x_test[0:(d_lin - input_dim), j]
        x_test[0:input_dim, j + 1] = x_test[0:input_dim, j] + W_out @ out_test
    return x_test


def compute_testing_prediction_with_params(linear_vector: np.ndarray, feature_obj: FeatureVector, W_out: np.ndarray, warmup_train: int, test_pts: int, input_dim: int, params: np.ndarray) -> np.ndarray:
    """
    Compute testing prediction with additional control parameters.
    
    :param linear_vector: Linear feature vector.
    :param feature_obj: Instance of FeatureVector.
    :param W_out: Output weights from training.
    :param warmup_train: Total warmup and training points.
    :param test_pts: Number of testing points.
    :param input_dim: Input dimension.
    :param params: Additional control parameters as a numpy array.
    :return: Predicted testing data with control parameters.
    """
    d_lin = feature_obj.linear_feature_vector_size()
    x_test = np.zeros((d_lin, test_pts))
    x_test[:, 0] = linear_vector[:, warmup_train - 1]
    
    for j in range(test_pts - 1):
        out_test = np.zeros(feature_obj.full_feature_vector_size() + len(params))
        cte = feature_obj.model_parameters.constant()
        out_test[cte: d_lin + cte] = x_test[:, j]
        index_combinations = feature_obj.polynomial_combinations()
        cnt = 0
        for comb in index_combinations:
            prod = feature_obj.calc_polynomial_prod(comb, x_test, j)
            out_test[cte + d_lin + cnt] = prod
            cnt += 1

        # Append additional parameters at the end of the feature vector
        params_control_count = -1 * len(params)
        params_index = 0
        while params_control_count != 0:
            out_test[params_control_count] = params[params_index]
            params_index += 1
            params_control_count += 1
        
        x_test[input_dim:d_lin, j + 1] = x_test[0:(d_lin - input_dim), j]
        x_test[0:input_dim, j + 1] = x_test[0:input_dim, j] + W_out @ out_test
    return x_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, total_variance: float) -> Dict[str, Any]:
    """
    Calculate various performance metrics including RMSE, MAE, R2, MAPE, and NRMSE.
    
    :param y_true: True values.
    :param y_pred: Predicted values.
    :param total_variance: Total variance of the true data.
    :return: Dictionary with metrics.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    nrmse = np.sqrt(np.mean((y_true - y_pred) ** 2) / total_variance)
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE (%)": mape, "NRMSE": nrmse}
