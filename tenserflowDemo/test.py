import  tensorflow as tf
import  numpy as np
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

def test2():
    from tensorflow.contrib import learn
    x_text = ['This is a cat', 'This must be boy', 'This is a a dog']
    max_document_length =max([len(x.split(' ')) for x in x_text])
    # print(max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    vocab_dict = vocab_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    print(sorted_vocab)
    vocabulary = list(list(zip(*sorted_vocab))[0])
    print(vocabulary)

def test3():
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    iterator = dataset1.make_initializable_iterator()
    next_element = iterator.get_next()
    next_variy = tf.Variable(next_element)
    print(next_element)
    init_ops = tf.global_variables_initializer()
    with tf.Session() as session:
        result = session.run(init_ops)
        print(result)

test3()