import  tensorflow as tf

def tensorDemo1():
    matrix1 = tf.constant([[3.1,3.5]])
    matrix2 = tf.constant([[2.],[2.]])

    product = tf.matmul(matrix1,matrix2)


    with tf.Session() as session:
        result = session.run([product])
        print(result)

def tensorDemo2():
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input2,input3)
    mul = tf.multiply(input1,intermed)

    with tf.Session() as session:
        result = session.run([mul])
        print(result)

def tensorFeedDemo():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    mul = tf.multiply(input1,input2)

    with tf.Session() as session:
        result = session.run(mul,feed_dict={ input1:[7.0], input2:[8.0] })
        print(result)

tensorFeedDemo()