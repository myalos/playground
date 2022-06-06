import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from IPython import embed
import tensorflow.contrib.graph_editor as ge

housing = fetch_california_housing()
# housing.data的shape 是 20640, 8
m, n = housing.data.shape
housing_data_scaled = StandardScaler().fit_transform(housing.data)

scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), housing_data_scaled]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype = tf.float32, name = "X")
y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name = "y")
# 要训练的，就是要变化的就是变量
theta = tf.Variable(tf.random_uniform((n + 1, 1), -1.0, 1.0), name = "theta")
y_pred = tf.matmul(X, theta, name = "predictions")
with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name = "mse")
print(error.op.name)
print(mse.op.name)
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
gradients = tf.gradients(mse, [theta])[0]
bwd_ops = ge.get_backward_walk_ops([mse.op], inclusive = True)
print(ge.get_backward_walk_ops([mse.op], inclusive = True))
print(ge.get_backward_walk_ops([mse.op], inclusive = False))
print(ge.get_forward_walk_ops([theta.op, X.op, y.op], inclusive = True))
print(ge.get_forward_walk_ops([theta.op, X.op, y.op], inclusive = True, within_ops = bwd_ops))
fwd_ops = ge.get_forward_walk_ops([theta.op, X.op, y.op], inclusive = True, within_ops = bwd_ops)
#print(fwd_ops[0].inputs)
#print(fwd_ops[1].inputs)
#print(fwd_ops[2].inputs)
fwd_ops1 = [op for op in fwd_ops if not op.inputs]
print(fwd_ops1)
print(fwd_ops1[0], fwd_ops1[0].inputs)
print(fwd_ops)
fwd_ops = [op for op in fwd_ops if op.inputs]
print(fwd_ops)


training_op = tf.assign(theta, theta - learning_rate * gradients)

#init = tf.global_variables_initializer()
#saver = tf.train.Saver()
#mse_summary = tf.summary.scalar('MSE', mse)
#mse_summary2 = tf.summary.scalar('MSE2', mse)

#file_writer = tf.summary.FileWriter('log')

#with tf.Session() as sess:
#    epoch = 0
#    sess.run(init)
#    #saver.restore(sess, "model.ckpt")
#    while epoch <= n_epochs:
#        if epoch % 100 == 0:
#            summary_str = mse_summary.eval()
#            summary_str2 = mse_summary2.eval()
#            file_writer.add_summary(summary_str, epoch)
#            file_writer.add_summary(summary_str2, epoch)
#            #print("Epoch", epoch, "MSE =", mse.eval())
#        #if epoch == 100:
#        #    saver = saver.save(sess, "model.ckpt")
#        #    break
#        sess.run(training_op)
#        epoch += 1
#    best_theta = theta.eval()
#print(best_theta)

