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
