import json
import numpy as np


class Perceptron:
    def __init__(self, layers: list[int] = None, filename: str = None):
        """Initialize the perceptron with the number of neurons in the layers (weights will be random) or with a file"""
        if filename is not None:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)

            self.__layers_count = len(data['weights']) + 1
            self.__weights = [np.array(weight) for weight in data['weights']]
            self.__biases = [np.array(bias) for bias in data['biases']]

        elif layers is not None:
            self.__layers_count = len(layers)
            self.__weights = []
            self.__biases = []

            for i in range(len(layers) - 1):
                self.__weights.append(np.random.randn(layers[i], layers[i + 1]))
                self.__biases.append(np.random.randn(layers[i + 1]))

        self.__activation_function = lambda x: np.maximum(x, 0)
        self.__activation_function_derivative = lambda x: (x >= 0).astype(float)
        pass

    def set_activation_function(self, f: callable):
        """Set the activation function and calculate its derivative"""
        self.__activation_function = np.vectorize(f)
        self.set_activation_function_derivative(lambda x: self.__get_derivative(f, x))
        pass

    def set_activation_function_derivative(self, f: callable):
        """Set the derivative of the activation function"""
        self.__activation_function_derivative = np.vectorize(f)
        pass

    def inference(self, input_data: list[float]) -> list[float]:
        """Pass the data through the perceptron and return the result"""
        h = np.array(input_data)

        for i in range(self.__layers_count - 1):
            t = h @ self.__weights[i] + self.__biases[i]
            h = self.__activation_function(t)

        return h.tolist()[0]

    def train(self, input_data: list[float], target_data: list[float], learning_rate: float) -> float:
        """Change perceptron weights with training example. Returns square error"""
        # Forward
        h = []
        t = []
        w_errors = []
        b_errors = []
        h.append(np.matrix(input_data))

        for i in range(self.__layers_count - 1):
            t.append(h[i] @ self.__weights[i] + self.__biases[i])
            h.append(self.__activation_function(t[i]))

        # Back
        t_error = h[-1] - target_data
        error = sum([i ** 2 for i in t_error.tolist()[0]])

        for i in range(self.__layers_count - 2, -1, -1):
            w_errors.insert(0, h[i].T @ t_error)
            b_errors.insert(0, t_error)

            if i > 0:
                h_error = t_error @ self.__weights[i].T
                t_error = np.multiply(h_error, self.__activation_function_derivative(t[i - 1]))

        # Update weights
        for i in range(self.__layers_count - 1):
            self.__weights[i] = self.__weights[i] - learning_rate * w_errors[i]
            self.__biases[i] = self.__biases[i] - learning_rate * b_errors[i]

        return error

    @staticmethod
    def __get_derivative(f: callable, x: float) -> float:
        """Automatic calculation of the derivative of a function"""
        dx = 0.002
        y1 = f(x - 0.001)
        y2 = f(x + 0.001)
        dy = y2 - y1
        return dy / dx

    def save(self, filename: str):
        """Save weights and biases to JSON file"""
        weights = [weight.tolist() for weight in self.__weights]
        biases = [bias.tolist() for bias in self.__biases]
        data = {'weights': weights, 'biases': biases}
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        pass
