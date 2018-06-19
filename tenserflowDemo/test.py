import  tensorflow as tf
def test1():
    import numpy as np
    x_vals = np.arange(1,10,2,dtype=np.float32)
    my_array =  np.array([x_vals,x_vals+1])
    print(my_array)
    x_data = tf.placeholder(tf.float32)
    m_const = tf.constant(3.)
    my_product = tf.multiply(x_data,m_const)
    with tf.Session() as session:
        for x in x_vals:
            result = session.run(my_product,feed_dict={x_data:x})
            print(result)

test1()
