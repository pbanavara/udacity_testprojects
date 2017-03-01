import tensorflow as tf
import input_data
from tensorflow.contrib.layers import flatten

#Hyper parameters
num_classes = 10
learning_rate = 0.001
STEP_SIZE = 100

def train(loss, global_step):
    learning_rate = 1e-3
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    print("Loss inside Gradient descent :::", train_op)
    return train_op

def inference(x):
    """
    Builds the video convnet model and returns the logits
    """
    #conv1 
    with tf.variable_scope('conv1') as scope:
        #w_initializer = tf.truncated_normal_initializer(stddev=1e-4)
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        print("Initializer :::", w_initializer)
        #weights = tf.get_variable('weights', [5,5,3,64], initializer=tf.truncated_normal_initializer(stddev=1e-4))
        weights = tf.get_variable('weights', [5,5,3,64], initializer =
                                    w_initializer)
        conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding = 'SAME')
        #biases = tf.get_variable('biases', [64], initializer = tf.constant.initializer(0.0))
        b_initializer = tf.constant_initializer(0.1)
        biases = tf.get_variable("biases", [64], 
                        initializer = b_initializer)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name = scope.name)

    #pool1
    print ("Conv 1 Shape ::::", conv1.get_shape())
    pool1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1],
                            name = 'pool1', padding = 'SAME')
    print ("Pool 1 shape :::" , pool1.get_shape())
    #norm1
    norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75,
                        name = 'norm1')
    norm1 = flatten(norm1, scope = 'norm1')
    print("Norm 1 :::", norm1.get_shape())
    #Fully connected layer
    with tf.variable_scope('fc') as scope:
        weights = tf.get_variable("weights", [9216, 384], dtype = tf.float32,
                        initializer = tf.truncated_normal_initializer(stddev = 1e-4))
        biases = tf.get_variable('biases', [384], 
                    initializer = tf.constant_initializer(0.1))
        fc = tf.nn.relu(tf.matmul(norm1, weights) + biases, name = scope.name)
    
    #Final fully connected layer
    with tf.variable_scope('final') as scope:
        weights = tf.get_variable('weights', [384, num_classes], initializer=tf.truncated_normal_initializer(stddev = 1e-4))
        biases = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.1))
        final = tf.add(tf.matmul(fc, weights), biases, name = scope.name)
    return final

    """
    #conv2
    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape = [3,3,3,64], 
                            initializer = tf.random_normal_initializer())
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding = 'SAME')
        biases = _variable('biases', [64], tf.constant.initializer(0.1))
        pre_acivation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name = scope.name)
    """

def loss_function(logits, y):
    """
    Runs the loss function on the logits returned from inference and returns a
    loss tensor
    """
    y = tf.squeeze(tf.cast(y, tf.int64))
    print("Logits in loss shape ::::", logits.get_shape())
    one_hot_y = tf.one_hot(y, num_classes)
    print ("Labels in loss shape ::::", one_hot_y.get_shape())
    print("Logits in loss::::", logits)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits,
                    y,
                    )
    loss = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    print("Cross Entropy", loss)
    return loss

    
def main(argv = None):
    global_step = tf.Variable(STEP_SIZE, name='global_step', trainable=False)
    #x = tf.placeholder(tf.float32, [None, 24,24,3])
    #y = tf.placeholder(tf.int32, (None))
    with tf.Graph().as_default():
        images, labels = input_data.distorted_inputs('file_with_labels.txt', 5)
        logits = inference(images)
        loss = loss_function(logits, labels)
        train_op = train(loss, global_step)
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        for i in range(STEP_SIZE):
            _, l = sess.run([train_op, loss])
            print ("Loss ::::", l)
            #sess.run(loss, feed_dict = {x: images, y:labels})
        saver.save(sess, './karpathy_1')

if __name__ == "__main__":
    main()
