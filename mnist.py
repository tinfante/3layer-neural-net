#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot
from neuralNetwork import neuralNetwork


def read_lines(filename):
    with open(filename, 'r') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    return lines


def read_mnist(filename):
    data = [[int(col) for col in row.split(',')]
            for row in read_lines(filename)]
    return data


def draw_number(mnist_row):
    image_data = numpy.asfarray(mnist_row[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_data, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()


def scale_input(mnist_row):
    image_array = mnist_row[1:]
    return (numpy.asfarray(image_array) / 255.0 * 0.99) + 0.01


def target_output(output_nodes, mnist_row):
    image_class = mnist_row[0]
    targets = numpy.zeros(output_nodes) + 0.01
    targets[image_class] = 0.99
    return targets


def main():
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    epochs = 5
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes,
                      learning_rate)
    #train = read_mnist('MNIST_dataset/mnist_train_100.csv')
    train = read_mnist('MNIST_dataset/mnist_train.csv')
    for epoch in range(epochs):
        for row in train:
            inputs = scale_input(row)
            targets = target_output(output_nodes, row)
            n.train(inputs, targets)

    #test = read_mnist('MNIST_dataset/mnist_test_10.csv')
    test = read_mnist('MNIST_dataset/mnist_test.csv')
    scores = []
    for row_num, row in enumerate(test):
        actual_class = row[0]
        inputs = scale_input(row)
        outputs = n.query(inputs)
        predicted_class = numpy.argmax(outputs)
        #print('ROW #:', row_num, '\t',
              #'CLASS:', actual_class, '\t',
              #'PREDICTED:', predicted_class)
        if actual_class == predicted_class:
            scores.append(1)
        else:
            scores.append(0)
    print('ACCURACY:', sum(scores) / len(scores))



if __name__ == '__main__':
    main()
