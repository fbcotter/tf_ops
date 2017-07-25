TF Layers Example
=================

After a little bit of looking at tf.layers, I have realized that the
functionality it implements is very good, but the documentation for it is quite
minimal (and leaves lots of gaps). However, looking at the `source`__ makes
things much clearer.

__ https://github.com/tensorflow/tensorflow/tree/003deb88b7fb015db86089c2a87b3044cad2c714/tensorflow/python/layers

In fact, there is quite a bit of functionality that is not available to you if
you use the functional api from tf.layers. If you instead use the underlying
classes:

- tensorflow.python.layers.convolutional.Conv2D
- tensorflow.python.layers.core.Dense
- tensorflow.python.layers.core.Dropout
- tensorflow.python.layers.normalization.BatchNormalization

you can get access to lots of helpful properties.

Convolution
-----------

For example, let us define a convolutional layer like so:

.. code:: python

    import tensorflow as tf, numpy as np
    from tensorflow.python.ops import init_ops
    from tensorflow.python.layers import convolutional
    x = 255 * np.random.rand(1, 50, 50, 3).astype(np.float32)
    v = tf.Variable(x)
    # Use glorot initialization
    init = init_ops.VarianceScaling(scale=1.0, mode='fan_out')
    # Use l2 regularization
    reg = tf.nn.l2_loss
    # Create an object representing the layer
    conv_layer = convolutional.Conv2D(
        out_filters, kernel_size=3, padding='same',
        kernel_initializer=init, kernel_regularizer=reg, name='conv')
    # Now get the outputs
    y = conv_layer.apply(x)

Now, we may want to get the weights that were defined to add some variable
summaries, or maybe we want to inspect the losses. Now we can do so by looking
at the properties of the `conv_layer`:

.. code:: python

    weights = conv_layer.trainable_weights
    variables = conv_layer.variables
    loss = conv_layer.losses

Batch Norm 
----------

I wanted to include an example of batch norm, as there are a few things to be
careful about. In particular, the apply method has a parameter `training`. We
can see the importance of this with an example:

.. code:: python

    import tensorflow as tf, numpy as np
    from tensorflow.python.ops import init_ops
    from tensorflow.python.layers import normalization
    x = 255 * np.random.rand(50, 50, 3).astype(np.float32)
    v = tf.Variable(x)

    bn_layer1 = normalization.BatchNormalization(name='bn1')
    bn_layer2 = normalization.BatchNormalization(name='bn2')
    y1 = bn_layer1.apply(v, training=True)
    y2 = bn_layer2.apply(v, training=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    y1_n, y2_n = sess.run([y1, y2])

    print('Input mean and std: {:.2f}, {:.2f}'.format(np.mean(x), np.std(x)))
    print('y1 mean and std: {:.2f}, {:.2f}'.format(np.mean(y1_n), np.std(y1_n)))
    print('y2 mean and std: {:.2f}, {:.2f}'.format(np.mean(y2_n), np.std(y2_n)))

Will have output::

    Input mean and std: 126.41, 74.26
    y1 mean and std: -0.00, 1.00
    y2 mean and std: 126.34, 74.22

This is because batch norm will subtract the batch mean and divide by the batch
standard deviation during training time to approximate an estimate on the
population mean and standard deviation. In this case we only had one example,
so that meant it got zero centred.

Similarly, for test time, the batch norm layer will want to subtract the
population mean and divide by the population standard deviation. When we start
training, these values are initialized to 0 and 1 respectively. When training, 
the moving_mean and moving_variance need to be updated.
By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
need to be added as a dependency to the `train_op`. For example:

.. code:: python

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

