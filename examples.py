import tensorflow as tf
import random
import math

# What is TensorFlow?
#
# - Machine learning frame work
# - Define a graph of operations
# - Define tunable variables (weights/biases) in the graph
# - Run your graph with input parameters
# - Use built-in error functions + optimizers to modify the variables in your graph


################################################################################################
# Graph of operations
#
# You define your model in TensorFlow using a graph of operations.
# Nodes: Multi-dimensional arrays (called Tensors)
# Edges: Built in (or custom) operations

a = tf.constant(1, dtype=tf.float32, shape=[2, 4, 3])   # Explicitly defined Tensor
b = tf.constant(2, dtype=tf.float32, shape=[4, 3])   # Explicitly defined Tensor

c = tf.add(a, b)                                        # Tensor resulting from operation

# TensorFlow has a very large variety of operations

# Note that as I try and print(c), I don't see it's value. In fact, in the debugger,
# the c Tensor object has no attribute for it's value!
# This is because when you code in TensorFlow, you are 'defining' a graph of operations,
# but not 'running' that graph.
# To run the graph, we need to instantiate a session and pass our graph to it. Then,
# TensorFlow will execute our session on it's C++ backend


################################################################################################
# Sessions

sess = tf.Session()
result = sess.run([c])                                  # Pass the leaf nodes of the graph you want to run

# You can also run multiple leaf nodes at once and get multiple results
d = tf.multiply(c, b)
result1, result2 = sess.run([c, d])

# General idea:
# 1. Define the graph for your model
# 2. Run it on a few concatenated samples of your data (batch) with sess.run()
# 3. Compare the results to the known answers (labels)
# 4. Modify your model's variables according to some learning algorithm (SGD)
# 5. Repeat steps 2-4 over all of your data several times using a for-loop

# Note: in practice, the error function and optimizer from steps 3-4 are also part of your computation graph

# Currently, nothing in the small graph that we defined has no variables. Let's change that!


################################################################################################
# Training a small model
#
# To illustrate how training works, we're going to make a very small neural net that learns how to
# predict the magnitude of a 3-dimensional vector.
# (i.e. we are trying to learn y = sqrt(x[0]**2 + x[1]**2 + x[2]**2)

# First, to get this toy example working, we need to generate some example data.
# Note: This step isn't necessary for ordinary deep learning, since we already have the data.
inputs = [[random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3)] for i in range(10000)]
labels = [math.sqrt(input[0]**2 + input[1]**2 + input[2]**2) for input in inputs]

# Let's assume that when training, we will be using a batch size of 4 (passing 4 samples to the model at a time),
# and that we will go across all of our data 3 times (3 epochs)
epochs = 20
batch_size = 4

####################################
# Define the model

# Our model will consist of 2 fully-connected layers, outputing 100 units and 1 unit respectively
input_size = 3
layer1_size = 100
layer2_size = 1     # The output

input_placeholder = tf.placeholder(tf.float32, shape=[None, 3])     # Specify when calling sess.run()

# Define all our trainable variable tensors
weights_layer1 = tf.Variable(initial_value=tf.random_normal(shape=[input_size, layer1_size]))
biases_layer1 = tf.Variable(initial_value=tf.random_normal(shape=[layer1_size]))
weights_layer2 = tf.Variable(initial_value=tf.random_normal(shape=[layer1_size, layer2_size]))

# Connect everything together (using x for the current units)
x = input_placeholder
x = tf.matmul(x, weights_layer1) + biases_layer1        # + is equivalent to tf.add()
x = tf.nn.relu(x)                                       # Non-linearity applied on activations (relu most popular)
prediction = tf.matmul(x, weights_layer2)

# Okay, now before training, let's just see what our model initialized with random weights outputs for the
# first 4 samples of our data
sess.run(tf.global_variables_initializer())             # This initializes any variables in the model
test_outputs = sess.run(prediction, feed_dict={input_placeholder: inputs[:4]})

####################################
# Train the model

# We need to define a loss function that we use to compare our predictions to our labels.
# Luckily, TensorFlow has a lot built in
label_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
loss = tf.losses.mean_squared_error(label_placeholder, prediction)

# Next, we define our optimizer. Each time it runs in a session, it updates the Variables that make up our graph
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
update = optimizer.minimize(loss)

# And now let's actually train
sess.run(tf.global_variables_initializer())
for epoch in range(1, epochs+1):
    for i in range(0, len(inputs), batch_size):
        input_batch = inputs[i:i+batch_size]
        label_batch = labels[i:i+batch_size]

        _, l = sess.run([update, loss],
                        feed_dict={input_placeholder: input_batch,
                                   label_placeholder: label_batch})

        if i % 100:
            print(l)


# And there we have it! We've trained a model. But TensorFlow actually has a more abstracted API for deep
# learning to help us make our model definition cleaner

################################################################################################
# Using tf.layers
#
# With what we had before, we had to explicitly define the weight/bias Variables and the matrix multiplication
# connecting them. tf.layers abstracts this process

# Let's redefine our model with tf.layers
x = input_placeholder
x = tf.layers.dense(inputs=x,
                    units=layer1_size,
                    activation=tf.nn.relu)
prediction = tf.layers.dense(inputs=x,
                             units=layer2_size,
                             use_bias=False)

# Now we can train our model as usual. The Variables were defined implicitly inside the call to 'dense', so
# we don't have to worry about them. It may not look much cleaner now, but the advantages become very apparent
# when working with more complex layers

# 'dense', as well as the other layers, have many optional parameters, such as how you want to initialize your weights
