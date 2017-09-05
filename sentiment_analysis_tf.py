import tensorflow as tf
from sentiment_analysis import classification_feature_creator
import numpy as np

training_input, training_output, test_input, test_output = classification_feature_creator('pos.txt', 'neg.txt')

number_nodes_h1 = 500
number_nodes_h2 = 500
number_nodes_h3 = 500

number_classes = 2
batch_size = 100

# len of the training input is the number of words in our lexicon
# 'x' and 'y' are the placeholders used to store the input and output data
x = tf.placeholder('float', [None, len(training_input[0])])
y = tf.placeholder('float')

# neural network placeholder
def neural_network_model(data):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([len(training_input[0]), number_nodes_h1])),
                      'biases': tf.Variable(tf.random_normal([number_nodes_h1]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([number_nodes_h1, number_nodes_h2])),
                      'biases': tf.Variable(tf.random_normal([number_nodes_h2]))}
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([number_nodes_h2, number_nodes_h3])),
                      'biases': tf.Variable(tf.random_normal([number_nodes_h3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([number_nodes_h3, number_classes])),
                    'biases': tf.Variable(tf.random_normal([number_classes]))} 
    layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(hidden_layer_1['weights'], hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(hidden_layer_2['weights'], hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)
    output = tf.add(tf.matmul(hidden_layer_3['weights'], output_layer['weights']), output_layer['biases']) 
    return output

def model_trainer(x):
    predicted_value = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_value, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    number_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(number_epochs):
            loss = 0
            iterator = 0
            while iterator < len(training_input):
                start = iterator
                end = iterator + batch_size
                batch_x, batch_y = np.array(training_input[start:end]), np.array(training_output[start:end])
                temp, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y}) 
                loss += c
                iterator += batch_size
            print ('Epoch', epoch + 1, 'completed out of', number_epochs, 'loss:', loss)
        correct = tf.equal(tf.argmax(predicted_value, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print ('Accuracy :', accuracy.eval({x: test_x, y: test_y}))

model_trainer(x)
