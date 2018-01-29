import tensorflow as tf
from dataset import MnistDataset

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('saved_models/model.ckpt.meta')      # Don't have to recreate the entire graph
    saver.restore(sess, 'saved_models/model.ckpt')                          # Restore all graph variables

    model = tf.get_collection('model')[0]
    inputs = tf.get_collection('model_inputs')[0]

    test_inputs = ['data/test/img_1.jpg',
                   'data/test/img_2.jpg',
                   'data/test/img_3.jpg']
    test_inputs = [MnistDataset.read_image(input) for input in test_inputs]
    predictions = sess.run(model,
                           feed_dict={inputs: test_inputs})
    print(predictions)
