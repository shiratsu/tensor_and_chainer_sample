import tensorflow as tf
import numpy

SCORE_SIZE = 33
HIDDEN_UNIT_SIZE = 32
TRAIN_DATA_SIZE = 90

raw_input = numpy.loadtxt(open("data/input.csv"), delimiter=",")
[salary, score]  = numpy.hsplit(raw_input, [1])

[salary_train, salary_test] = numpy.vsplit(salary, [TRAIN_DATA_SIZE])
[score_train, score_test] = numpy.vsplit(score, [TRAIN_DATA_SIZE])

def inference(score_placeholder):
  with tf.name_scope('hidden1') as scope:
    hidden1_weight = tf.Variable(tf.truncated_normal([SCORE_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="hidden1_weight")
    hidden1_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="hidden1_bias")
    hidden1_output = tf.nn.relu(tf.matmul(score_placeholder, hidden1_weight) + hidden1_bias)
  with tf.name_scope('output') as scope:
    output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, 1], stddev=0.1), name="output_weight")
    output_bias = tf.Variable(tf.constant(0.1, shape=[1]), name="output_bias")
    output = tf.matmul(hidden1_output, output_weight) + output_bias
  return tf.nn.l2_normalize(output, 0)

def loss(output, salary_placeholder, loss_label_placeholder):
  with tf.name_scope('loss') as scope:
    loss = tf.nn.l2_loss(output - tf.nn.l2_normalize(salary_placeholder, 0))
    tf.scalar_summary(loss_label_placeholder, loss)
  return loss

def training(loss):
  with tf.name_scope('training') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step


with tf.Graph().as_default():
  salary_placeholder = tf.placeholder("float", [None, 1], name="salary_placeholder")
  score_placeholder = tf.placeholder("float", [None, SCORE_SIZE], name="score_placeholder")
  loss_label_placeholder = tf.placeholder("string", name="loss_label_placeholder")

  feed_dict_train={
    salary_placeholder: salary_train,
    score_placeholder: score_train,
    loss_label_placeholder: "loss_train"
  }

  feed_dict_test={
    salary_placeholder: salary_test,
    score_placeholder: score_test,
    loss_label_placeholder: "loss_test"
  }

  output = inference(score_placeholder)
  loss = loss(output, salary_placeholder, loss_label_placeholder)
  training_op = training(loss)

  summary_op = tf.merge_all_summaries()

  init = tf.initialize_all_variables()
  
  best_loss = float("inf")
  print("------------init-----------------")

  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)
    sess.run(init)

    for step in range(10000):
      sess.run(training_op, feed_dict=feed_dict_train)
      loss_test = sess.run(loss, feed_dict=feed_dict_test)
      print("------------best-----------------")
      print(best_loss)
      print("------------loss-----------------")
      print(loss_test)
      if loss_test < best_loss:
        best_loss = loss_test
        best_match = sess.run(output, feed_dict=feed_dict_test)
      if step % 100 == 0:
        summary_str = sess.run(summary_op, feed_dict=feed_dict_test)
        summary_str += sess.run(summary_op, feed_dict=feed_dict_train)
        summary_writer.add_summary(summary_str, step)

    print(sess.run(tf.nn.l2_normalize(salary_placeholder, 0), feed_dict=feed_dict_test))
    print(best_match)
