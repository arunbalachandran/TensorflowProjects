import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets(os.getenv('TEMP_TF_DATA'), one_hot=True)

number_h1_nodes = 100
number_h2_nodes = 100
number_h3_nodes = 100

# we have 10 classification classes for the mnist dataset
number_classes = 10
batch_size = 100  # train in batches of 100 images at a time

# convert the pixel data into a single row and send for processing
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def nn_model(data):
    # initialize with random data
    # each of the 784 input data values will flow to a node with a different weight
    # one thing to ask yourself would be, if the weights from a node to the other nodes is random
    # or, are all possible combinations of weightings generated randomly
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, number_h1_nodes])),
                      'biases': tf.Variable(tf.random_normal([number_h1_nodes]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([number_h1_nodes, number_h2_nodes])),
                      'biases': tf.Variable(tf.random_normal([number_h2_nodes]))}
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([number_h2_nodes, number_h3_nodes])),
                      'biases': tf.Variable(tf.random_normal([number_h3_nodes]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([number_h3_nodes, number_classes])),
                    'biases': tf.Variable(tf.random_normal([number_classes]))}
    # now write the formulas and also, apply the activation function on each layer
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output

def train_nn(x):
    prediction = nn_model(x)
    # now decide the cost function to be used
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # select the optimization algorithm to be used, and minimize the cost (i.e. the distance from predicted value)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    number_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(number_epochs):
            epoch_loss = 0
            for temp in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                temp_1, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', number_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print ('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_nn(x)
