# Perceptron
Adds a Perceptron class that allows you to create, train, save, and load multilayer perceptrons.
## Usage:
The perceptron constructor allows you to determine the number of neurons on each layer. In this example, a perceptron with 4 layers is created: 3 neurons in the input layer, 4 neurons in the first hidden layer, 5 neurons in the second hidden layer, 2 neurons in the output layer.
```python
from perceptron import Perceptron


p = Perceptron([3, 4, 5, 2])
```
