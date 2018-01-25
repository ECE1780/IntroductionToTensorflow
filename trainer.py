import tensorflow as tf
from models.basic_model import BasicModel
from models.convolutional_model import ConvModel
from dataset import MnistDataset

epochs = 5
batch_size = 4

train_set = MnistDataset('data/train')
n_batches = len(train_set) // batch_size

model = BasicModel(resolution=[28, 28], channels=1)
# model = ConvModel(resolution=[28, 28], channels=1)

saver = tf.train.Saver()        # We use this to save the model. Instantiate it after all Variables have been created

label_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 10])
loss = tf.losses.softmax_cross_entropy(label_placeholder, model.predictions)
update = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

top_predictions = tf.argmax(model.predictions, axis=1)      # probabilities -> top prediction
top_labels = tf.argmax(label_placeholder, axis=1)           # one_hot -> number
correct = tf.equal(top_predictions, top_labels)             # bool Tensor
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))     # Average correct guesses

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        print('Starting epoch %d' % epoch)
        total_epoch_loss = 0

        for i in range(n_batches):
            images, labels = train_set.sample(batch_size)

            _, l, a = sess.run([update, loss, accuracy],
                               feed_dict={model.input_placeholder: images,
                                          label_placeholder: labels})
            total_epoch_loss += l

            if i % 100 == 0:
                print('[%d / %d] Accuracy: %.2f%%     Loss: %f' % (i+1, n_batches, a*100, l))

        print('Average epoch loss: %f\n' % (total_epoch_loss / n_batches))

    tf.add_to_collection('model', model.predictions)      # Specify the graph nodes that we want to use later
    tf.add_to_collection('model_inputs', model.input_placeholder)
    saver.save(sess, './saved_models/model')                # Save the entire graph and all Variables

