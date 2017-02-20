import pickle
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# TODO: Load traffic signs data.
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']
print(X_train.shape)

# TODO: Split data into training and validation sets.
X_train, X_validation, y_train, y_validation = train_test_split(X_train,
y_train, stratify = y_train)

print(X_train.shape)
print(X_validation.shape)

# TODO: Define placeholders and resize operation.
n_classes = 43
learning_rate = 1e-3
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

resized = tf.image.resize_images(x, (227,227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], n_classes)
fc8w = tf.Variable(tf.random_normal(shape, name = "fc8w"))
fc8b = tf.zeros([n_classes], name = "fc8b")
logits = tf.nn.xw_plus_b(fc7, fc8w, fc8b)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
y))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#Prediction
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
print("Correct prediction", correct_prediction)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation,
                            feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# TODO: Train and evaluate the feature extraction model.
init = tf.global_variables_initializer()
EPOCHS = 10
BATCH_SIZE = 128
with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_train)
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(train_op, feed_dict = {x: batch_x, y: batch_y})
            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
