import  tensorflow as tf

with tf.Session() as session:
    result = session.run(tf.nn.relu([-3.,3.,10.]))
    print(result)