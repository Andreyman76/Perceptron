# Perceptron
Adds a *Perceptron* class that allows you to create, train, save, and load multilayer perceptrons.
## Usage:
The perceptron constructor allows you to determine the number of neurons on each layer. In this example, a perceptron with 4 layers is created: 3 neurons in the input layer, 4 neurons in the first hidden layer, 5 neurons in the second hidden layer, 2 neurons in the output layer.
```python
from perceptron import Perceptron


p = Perceptron([3, 4, 5, 2])
```
**ReLu** is used as the default activation function. You can change the activation function yourself using the *set_activation_function* method (the derivative will be calculated automatically). If you are not satisfied with the automatic calculation of the derivative, then explicitly set the derivative of the activation function using the *set_activation_function_derivative* method.
```python
import math
from perceptron import Perceptron


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1.0 - sigmoid(x))


p = Perceptron([3, 4, 2])
p.set_activation_function(sigmoid)
p.set_activation_function_derivative(sigmoid_derivative)
```
The *train* method is used to train the perceptron. This method accepts 2 lists of float values (an input and an expected output) and learning rate (float). The return value is the squared error.
```python
from perceptron import Perceptron


data = [1.0, 0.0, 1.0]
target = [1.0, 0.0]

p = Perceptron([3, 4, 2])
error = p.train(data, target, 0.001)
```
The *inference* method is used to pass data through the perceptron and get a prediction.
```python
from perceptron import Perceptron


data = [1.0, 0.0, 1.0]

p = Perceptron([3, 4, 2])
prediction = p.inference(data)
```
You can save perceptron weights with the *save* method. Perceptron weights are loaded using the constructor.
```python
from perceptron import Perceptron


p = Perceptron(filename='iris.json')  # Loading weights from a file
predict = p.inference([5.9, 3.2, 4.8, 1.8])
print(predict)
p.save('iris1.json')  # Saving weights to a file
```
An example demonstrating how to work with a *Perceptron* class.
```python
from sklearn import datasets
from perceptron import Perceptron


def index_of_max(x: list[float]) -> int:
    """Returns the index of the largest element in the list"""
    m = x[0]
    index = 0

    for i in range(1, len(x)):
        if x[i] > m:
            m = x[i]
            index = i

    return index


def train_epoch(p: Perceptron, data, target, learning_rate):
    """Train a perceptron during a training epoch. Returns the root mean square error"""
    length = len(data)
    error = 0
    for i in range(length):
        error += p.train(data[i], target[i], learning_rate)

    return error / length


def test(p: Perceptron, data, target):
    """Pass the test data through the perceptron and return the percentage of successful predictions"""
    length = len(data)
    incorrect = 0
    for i in range(length):
        result = p.inference(data[i])
        if index_of_max(result) != index_of_max(target[i]):
            print(f'Incorrect at {data[i]}. Expected {target[i]} but given {result}')
            incorrect += 1

    return (1.0 - incorrect / length) * 100


def main():
    # Use dataset from sklearn
    iris = datasets.load_iris()
    data = iris['data']
    # Convert target data from class indices to a vector of zeros and ones
    target = [[1.0 if i == j else 0.0 for j in range(3)] for i in iris['target']]

    p = Perceptron([4, 16, 8, 3])  # Comment it out if you are retraining the perceptron
    #p = Perceptron(filename='iris.json')  # Comment it out if you are creating a perceptron

    for epoch in range(1001):
        e = train_epoch(p, data, target, 0.001)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, error = {e}')

    print()
    correct = test(p, data, target)
    print()
    print(f'Correct: {correct}%')
    p.save('iris.json')
    pass


if __name__ == '__main__':
    main()
```
## Output:
```
Epoch 0, error = 3.1001741358077406
Epoch 100, error = 0.06595243920865809
Epoch 200, error = 0.055889025607961856
Epoch 300, error = 0.05230927347591407
Epoch 400, error = 0.04924466671865873
Epoch 500, error = 0.047051451291890006
Epoch 600, error = 0.045465108637177995
Epoch 700, error = 0.04547831786534051
Epoch 800, error = 0.045782420298396825
Epoch 900, error = 0.04446175696869583
Epoch 1000, error = 0.043669737356606175

Incorrect at [5.6 3.  4.5 1.5]. Expected [0.0, 1.0, 0.0] but given [0.0, 0.3030266359382361, 0.369799835205569]
Incorrect at [6.2 2.2 4.5 1.5]. Expected [0.0, 1.0, 0.0] but given [0.0, 0.11926570498226069, 0.836632165755299]
Incorrect at [5.9 3.2 4.8 1.8]. Expected [0.0, 1.0, 0.0] but given [0.0, 0.15704419717143805, 0.8556734488600999]
Incorrect at [6.3 2.5 4.9 1.5]. Expected [0.0, 1.0, 0.0] but given [0.0, 0.029447218893686677, 0.9839858593253279]
Incorrect at [6.7 3.  5.  1.7]. Expected [0.0, 1.0, 0.0] but given [0.0, 0.18692689403289167, 0.6357987731316606]
Incorrect at [5.7 2.6 3.5 1. ]. Expected [0.0, 1.0, 0.0] but given [0.0, 0.0, 0.0]
Incorrect at [6.  2.7 5.1 1.6]. Expected [0.0, 1.0, 0.0] but given [0.0, 0.00853171199720898, 1.0295117785259422]
Incorrect at [5.4 3.  4.5 1.5]. Expected [0.0, 1.0, 0.0] but given [0.0, 0.28412248931888057, 0.4221333693025614]

Correct: 94.66666666666667%
```