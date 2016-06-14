import tensorflow as tf
import numpy as np
import tensorflow as tf
import dataset

def train():
  sess = tf.InteractiveSession()
  with tf.name_scope('input'):
    x1 = tf.placeholder(tf.float32, [None, 6], name='x1')
    x2 = tf.placeholder(tf.float32, [None, 6], name='x2')
    y_ = tf.placeholder(tf.float32, [None], name='y')

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    #initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def layer((x1, x2), input_dim, output_dim, layer_name, act=tf.nn.relu):
    W = weight_variable([input_dim, output_dim])
    b = bias_variable([output_dim])
    p1, p2 = tf.matmul(x1, W) + b, tf.matmul(x2, W) + b
    o1, o2 = act(p1), act(p2)
    return (o1, o2)

  hidden1 = layer((x1, x2), 6, 1000, 'layer1')
  #hidden2 = layer(hidden1, 500,500, 'layer2')
  y1, y2 = layer(hidden1, 1000, 20, 'layer3')

  yy = tf.reduce_mean(y1, 1)
  y = tf.reduce_mean(y1 - y2, 1)
  cost = tf.reduce_mean(-y_ * y + tf.log(1 + tf.exp(y)))

  top = tf.nn.top_k(tf.reduce_mean(y1, 1), tf.shape(y1)[0], sorted=True)[1]
  #bad_10 = tf.nn.top_k(-tf.reduce_mean(y1, 1), 10, sorted=True)[1]

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.005).minimize(cost)

  with tf.name_scope('test'):
    miss = tf.logical_xor(y >= 0, y_ >= 0.5)
    accuracy = tf.reduce_mean(1. - tf.cast(miss, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

  def feed_dict(train):
    if train:
      x1s, x2s, ys, _ = dataset.get_train_data()
    else:
      x1s, x2s, ys, _ = dataset.get_test_data()
    return {x1 : x1s, x2 : x2s, y_ : ys}

  def model_data(token=''):
    data, _ = dataset.load_model_data(token)
    return {x1 : data}

  tf.initialize_all_variables().run()
  
  #fout = open('out.txt', 'w')
  for i in range(20):
    if i % 10 == 0 or i < 50:
      train_acc, res = sess.run([accuracy, y], feed_dict=feed_dict(True))
      test_acc,  res = sess.run([accuracy, y1], feed_dict=feed_dict(False))
      ys = [sum(x) / len(x) for x in res]
      print 'Accuracy at step {}: is {}, {}'.format(i, train_acc, test_acc)
      #print >>fout, '{}, {}'.format(train_acc, test_acc)
    sess.run([train_step], feed_dict=feed_dict(True))
  print 'Linear Accuracy: ', dataset.linear_acc(*(dataset.get_test_data()[:3]))

  model_name = dataset.token
  if model_name == '':
    return
  tops, ys = sess.run([top, yy], feed_dict=model_data(model_name))
  names = dataset.load_model_data(model_name)[1]
  print tops
  #n = len(names)
  #print '\n'.join([names[x] for x in tops])
  #print names[tops[0]], ys[tops[0]]
  #print dataset.linear_best(*dataset.load_model_data(model_name))
  #print names[tops[n / 4]], ys[tops[n / 4]]
  #print names[tops[n / 2]], ys[tops[n / 2]]
  #print names[tops[-n / 4]], ys[tops[- n / 4]]
  #print names[tops[-1]], ys[tops[-1]]
  #names = dataset.get_test_data()[3]
  #print 'good results'
  #print '\n'.join(names[x][0] for x in tops)
  #print 'bad results:'
  #print '\n'.join(names[x][0] for x in bads)

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()

  

          
