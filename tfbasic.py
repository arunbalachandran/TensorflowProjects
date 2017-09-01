import tensorflow as tf

# this is the computation graph
# define constants
x1 = tf.constant(12)
x2 = tf.constant(24)
# multiply the numbers using the mul function
result = tf.multiply(x1, x2)

# now we define the session which can modify the computation graph
with tf.Session() as sess:
    output = sess.run(result)

# output is a python variable so it can still be accessed
print (output)
