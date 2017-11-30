""" A collection of helper tf functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
from tensorflow.python.layers import convolutional, normalization
from tensorflow.python.ops import init_ops
from tf_ops.wave_ops import lazy_wavelet, lazy_wavelet_inv


def build_optimizer(opt_method, lr, loss, max_gradient_norm=None,
                    global_step=None, decay_steps=None):
    """ Build an optimizer and return a training op.

    Will also add checks for Nans in the gradient, and add some monitoring to
    tensorboard.

    Parameters
    ----------
    opt_method : str
        Either 'adam', 'sgd', or 'momentum'
    lr : float
        Learning rate for the optimizer
    loss : tf.Tensor
        Tensor containing the loss operation
    max_gradient_norm : float or None
        What the gradients should be clipped to. If None, no clipping used.
    global_step : tf.Variable or None
        Variable holding the global step
    decay_steps : int
        For sgd only. After how many steps to decay the learning rate.

    Returns
    -------
    train_op : tf op
        An op that can be run in a session to apply gradients to the trainable
        variables.
    """

    with tf.variable_scope('optimizer'):
        # Build the optimizer
        if opt_method == 'adam':
            print('Optimizing with Adam')
            opt = tf.train.AdamOptimizer(lr, epsilon=1e-6)
        elif opt_method == 'momentum':
            print('Optimizing with momentum')
            opt = tf.train.MomentumOptimizer(lr,
                                             momentum=0.9,
                                             use_nesterov=True)
        elif opt_method == 'sgd':
            # Decay the learning rate exponentially based on the number of steps
            if decay_steps is not None:
                lr = tf.train.exponential_decay(lr,
                                                global_step,
                                                decay_steps=500*100,
                                                decay_rate=0.1,
                                                staircase=True)
            opt = tf.train.GradientDescentOptimizer(lr)
            tf.summary.scalar('learning_rate', lr)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        # Collect the gradients and clip them
        if max_gradient_norm is not None:
            gradients, norm = tf.clip_by_global_norm(
                gradients, max_gradient_norm)
        else:
            norm = tf.sqrt(tf.reduce_sum(
                [tf.reduce_sum(g**2) for g in gradients if g is not None]))

        # Add checks on the gradients for Nans
        grad_check = [tf.check_numerics(g, '{} gradient nan'.format(p.name)) for
                      g, p in zip(gradients, params) if g is not None]

        grad_check.append(tf.check_numerics(norm, 'global clip val nan'))

        # Ensure the gradient check is done before applying the gradients
        with tf.control_dependencies(grad_check):
            train_op = opt.apply_gradients(
                zip(gradients, params), global_step=global_step)

    # Function to make a nicer name for display
    def strip_name(name):
        name = name.split(':')[0].split('/')
        if 'fwd' in name:
            name.remove('fwd')
        if 'batch_normalization' in name:
            name.remove('batch_normalization')
        return '/'.join(name)

    # Add some tensorboard logging
    with tf.variable_scope('grads'):
        tf.summary.scalar('all', norm)
        [tf.summary.scalar('{}'.format(strip_name(p.name)), tf.norm(g))
         for g, p in zip(gradients, params) if g is not None]

        # Add histograms for gradients.
        [tf.summary.histogram('{}'.format(strip_name(p.name)), g)
         for g, p in zip(gradients, params) if g is not None]

    return train_op


def variable_with_wd(name, shape, stddev=None, wd=None, norm=2):
    """ Helper to create an initialized variable with weight decay.

    Note that the variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified. Also will add summaries
    for this variable.

    Internally, it calls tf.get_variable, so you can use this to re-get already
    defined variables (so long as the reuse scope is set to true). If it
    re-fetches an already existing variable, it will not add regularization
    again.

    Parameters
    ----------
    name: str
        name of the variable
    shape: list of ints
        shape of the variable you want to create
    stddev: positive float or None
        standard deviation of a truncated Gaussian
    wd: positive float or None
        add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this variable.
    norm: positive float
        Which regularizer to apply. E.g. norm=2 uses L2 regularization, and
        norm=p adds :math:`wd \\times ||w||_{p}^{p}` to the
        REGULARIZATION_LOSSES. See :py:func:`real_reg`.

    Returns
    -------
    out : variable tensor
    """
    if stddev is None:
        stddev = get_xavier_stddev(shape, uniform=False)
    initializer = tf.truncated_normal_initializer(stddev=stddev)

    var_before = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    var = tf.get_variable(name, shape, dtype=tf.float32,
                          initializer=initializer)
    var_after = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    if len(var_before) != len(var_after):
        reg_loss = complex_reg(var, wd, norm)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg_loss)
        variable_summaries(var, name)

    return var


def variable_summaries(var, name='summaries'):
    """Attach a lot of summaries to a variable (for TensorBoard visualization).

    Parameters
    ----------
    var : :py:class:`tf.Tensor`
        variable for which you wish to create summaries
    name : str
        scope under which you want to add your summary ops
    """
    with tf.name_scope(name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)), name='stddev')
        tf.summary.scalar('standard_deviation', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def loss(labels, logits, one_hot=True, num_classes=None, λ=1):
    """ Compute sum of data + regularization losses.

    loss = data_loss + λ * reg_losses

    The regularization loss will sum over all the variables that already
    exist in the GraphKeys.REGULARIZATION_LOSSES.

    Parameters
    ----------
    labels : ndarray(dtype=float, ndim=(N,C))
        The vector of labels.
    one_hot : bool
        True if the labels input is one_hot.
    num_classes : int
        Needed if the labels aren't one-hot already.
    logits : tf.Variable
        Logit outputs from the neural net.
    λ : float
        Multiplier to use on all regularization losses. Be careful not
        to apply things twice, as all the functions in this module typically set
        regularization losses at a block level (for more fine control).
        For this reason it defaults to 1, but can be useful to set to some other
        value to get quick scaling of loss terms.

    Returns
    -------
    losses : tuple of (loss, data_loss, reg_loss)
        For optimization, only need to use the first element in the tuple. I
        return the other two for displaying purposes.
    """
    with tf.variable_scope('data_loss'):
        tf.summary.histogram('logits', logits)
        tf.summary.histogram('softmax', tf.nn.softmax(logits))
        if not one_hot:
            labels = tf.one_hot(labels, depth=num_classes, axis=-1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        data_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
    with tf.variable_scope('reg_loss'):
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.reduce_sum(reg_variables)

    with tf.variable_scope('loss'):
        loss = data_loss + λ*reg_term

    return loss, data_loss, reg_term


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Parameters
    ----------
    inputs: tf.Tensor
        A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: int
        The kernel to be used in the conv2d or max_pool2d operation.
        Should be a positive integer.
    data_format: str
        The input format ('channels_last' or 'channels_first').

    Returns
    -------
    y : tf.Tensor
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                               [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                               [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def _residual_core(x, filters, kernel_size=3, stride=1, train=True, wd=0.0,
                   bn_momentum=0.99, bn_epsilon=0.001):
    """ Core function of a residual unit.

    In -> conv -> bn -> relu -> conv

    Note that the normal residual layer has a batch norm and relu before the
    first conv. This is in the residual function which calls this.

    Parameters
    ----------
    x : tf tensor
        Input to be modified
    filters : int
        Number of output filters (will be used for all convolutions in the
        resnet core).
    kernel_size : int
        Size of the filter kernels
    stride : int
        Conv stride
    train : bool or tf boolean tensor
        Whether we are in the train phase or not. Can set to a tensorflow tensor
        so that it can be modified on the fly.
    wd : float
        Weight decay term for the convolutional weights
    bn_momentum : float
        The momentum for the batch normalization layers in the resnet
    bn_epsilon : float
        The epsilon for the batch normalization layers in the resnet
    """

    init = init_ops.VarianceScaling(scale=1.0, mode='fan_out')
    reg = lambda w: real_reg(w, wd, norm=2)
    bn_class = lambda name: normalization.BatchNormalization(
        name=name, momentum=bn_momentum, epsilon=bn_epsilon)
    conv_class = lambda name, stride: convolutional.Conv2D(
        filters, 3, (stride, stride), use_bias=False,
        padding=('SAME' if stride == 1 else 'VALID'),
        kernel_initializer=init, kernel_regularizer=reg, name=name)

    with tf.variable_scope('sub1'):
        # As we will do downsampling with strides, need to make sure the output
        # size is the correct format.
        if stride > 1:
            x = fixed_padding(x, kernel_size, 'channels_last')

        conv = conv_class('conv1', stride)
        x = conv.apply(x)

    with tf.variable_scope('sub2'):
        bn = bn_class('between_bn')
        x = bn.apply(x, training=train)
        x = tf.nn.relu(x)
        conv = conv_class('conv2', 1)
        x = conv.apply(x)

    return x


def residual(x, filters, kernel_size=3, stride=1, train=True, wd=0.0,
             bn_momentum=0.99, bn_epsilon=0.001, name='res'):
    """ Residual layer

    Uses the _residual_core function to create F(x), then adds x to it.

    Parameters
    ----------
    x : tf tensor
        Input to be modified
    filters : int
        Number of output filters (will be used for all convolutions in the
        resnet core).
    stride : int
        Conv stride
    train : bool or tf boolean tensor
        Whether we are in the train phase or not. Can set to a tensorflow tensor
        so that it can be modified on the fly.
    wd : float
        Weight decay term for the convolutional weights
    bn_momentum : float
        The momentum for the batch normalization layers in the resnet
    bn_epsilon : float
        The epsilon for the batch normalization layers in the resnet

    Notes
    -----
    When training, the moving_mean and moving_variance need to be updated. By
    default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need
    to be added as a dependency to the train_op. For example::

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
    """
    bn_class = lambda name: normalization.BatchNormalization(
        name=name, momentum=bn_momentum, epsilon=bn_epsilon)

    orig_x = x
    with tf.variable_scope(name):
        bn = bn_class('init_bn')
        x = bn.apply(x, training=train)
        x = tf.nn.relu(x)

        # The projection shortcut should come after the first batch norm and
        # ReLU since it performs a 1x1 convolution.
        if stride > 1:
            orig_x = tf.layers.conv2d(
                orig_x, filters=filters, strides=stride,
                kernel_size=1, padding='VALID', use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
                data_format='channels_last')

        x = _residual_core(x, filters, kernel_size, stride, train, wd,
                           bn_momentum, bn_epsilon)

        y = tf.add(x, orig_x)

    return y


def lift_residual_resample(x1, x2, filters, train=True, downsize=True,
                           wd=0.0001):
    """Define a Lifting Layer with resizing

    The P and the U blocks for this lifting layer are non-linear functions.
    These are the same form as the F(x) in a residual layer (i.e.  two
    convolutions). In block form, a lifting layer looks like this::

             _______________
            |               |
        x1->|---(+)---------|->d
            |    ^      |   |
            |    |      |   |
            |   ---    ---  |
            |  |-P |  | U | |
            |   ---    ---  |
            |    |      |   |
            |    |      v   |
        x2->|----------(+)--|->s
            |_______________|

    Parameters
    ----------
    x1 : tf tensor
        Input tensor 1
    x2 : tf tensor
        Input tensor 2
    filters : int
        Number of output channels for P*x2 and U*d
    train : bool or tf boolean tensor
        Whether we are in the train phase or not. Can set to a tensorflow tensor
        so that it can be modified on the fly.
    wd : float
        Weight decay term for the convolutional weights

    Returns
    -------
    d : tf tensor
        Detail coefficients
    s : tf tensor
        Scale coefficients
    """
    assert x1.get_shape().as_list() == x2.get_shape().as_list()

    # Split the two inputs
    if downsize:
        x1 = lazy_wavelet(x1)
        x2 = lazy_wavelet(x2)

    in_channels = x1.get_shape().as_list()[-1]
    if (filters % in_channels != 0):
        raise ValueError('Can only expand an image by an integer number ' +
                         'of its channels')

    # If the requested number of output channels is larger than the input,
    # stack the input to make them match.
    if filters != in_channels:
        # We've already checked that the remainder is 0 so can do integer div
        nstack = filters // in_channels
        x1 = tf.concat([x1]*nstack, axis=-1)

    d, s = lift_residual(x1, x2, train, wd)

    return d, s


def lift_residual_resample_inv(d, s, out_size, train=True, wd=0.0001):
    """Define a inverse Lifting Layer with resizing

    The P and the U blocks for this lifting layer are non-linear functions.
    These are the same form as the F(x) in a residual layer (i.e.  two
    convolutions). In block form, a lifting layer looks like this::

    We share the variables with the forward lifting.

    In block form, the inverse lifting layer looks like this (note the sign swap
    and flow direction reversal compared to the forward case)::

             _______________
            |               |
        x1<-|---(+)---------|<-d
            |    ^      |   |
            |    |      |   |
            |   ---    ---  |
            |  | P |  |-U | |
            |   ---    ---  |
            |    |      |   |
            |    |      v   |
        x2<-|----------(+)--|<-s
            |_______________|

    Parameters
    ----------
    d : tf tensor
        Input tensor 1
    s : tf tensor
        Input tensor 2
    out_size : list of ints
        Size of the resulting x1 tensors.
    train : bool or tf boolean tensor
        Whether we are in the train phase or not. Can set to a tensorflow tensor
        so that it can be modified on the fly.
    wd : float
        Weight decay term for the convolutional weights

    Returns
    -------
    x1 : tf tensor
        Reconstructed x1
    x2 : tf tensor
        Reconstructed x2
    """
    assert d.get_shape().as_list() == s.get_shape().as_list()
    x1, x2 = lift_residual_inv(d, s, train, wd)

    # Recombine the two outputs
    if out_size[-1] != x1.get_shape().as_list()[-1]:
        x1 = lazy_wavelet_inv(x1, out_size)
        x2 = lazy_wavelet_inv(x2, out_size)

    return x1, x2


def lift_residual(x1, x2, train=True, wd=0.0001):
    """Define a Lifting Layer

    The P and the U blocks for this lifting layer are non-linear functions.
    These are the same form as the F(x) in a residual layer (i.e.  two
    convolutions). In block form, a lifting layer looks like this::

             _______________
            |               |
        x1->|---(+)---------|->d
            |    ^      |   |
            |    |      |   |
            |   ---    ---  |
            |  |-P |  | U | |
            |   ---    ---  |
            |    |      |   |
            |    |      v   |
        x2->|----------(+)--|->s
            |_______________|

    Parameters
    ----------
    x1 : tf tensor
        Input tensor 1
    x2 : tf tensor
        Input tensor 2
    train : bool or tf boolean tensor
        Whether we are in the train phase or not. Can set to a tensorflow tensor
        so that it can be modified on the fly.
    wd : float
        Weight decay term for the convolutional weights

    Returns
    -------
    d : tf tensor
        Detail coefficients
    s : tf tensor
        Scale coefficients
    """
    filters = x1.get_shape().as_list()[-1]
    assert filters == x2.get_shape().as_list()[-1]
    with tf.variable_scope('P'):
        # Calculate d = x1 - Px2
        d = x1 - _residual_core(x2, filters, 1, train, wd)

    with tf.variable_scope('U'):
        # Calculate s = x2 + Ud
        s = x2 + _residual_core(d, filters, 1, train, wd)

    return d, s


def lift_residual_inv(d, s, train=True, wd=0.0001):
    """Define the inverse of a lifting layer

    We share the variables with the forward lifting.

    In block form, the inverse lifting layer looks like this (note the sign swap
    and flow direction reversal compared to the forward case)::

             _______________
            |               |
        x1<-|---(+)---------|<-d
            |    ^      |   |
            |    |      |   |
            |   ---    ---  |
            |  | P |  |-U | |
            |   ---    ---  |
            |    |      |   |
            |    |      v   |
        x2<-|----------(+)--|<-s
            |_______________|


    Parameters
    ----------
    d : tf tensor
        Input tensor 1
    s : tf tensor
        Input tensor 2
    filters : int
        Number of output channels for Px2 and Ud
    train : bool or tf boolean tensor
        Whether we are in the train phase or not. Can set to a tensorflow tensor
        so that it can be modified on the fly.
    wd : float
        Weight decay term for the convolutional weights

    Returns
    -------
    x1 : tf tensor
        Reconstructed x1
    x2 : tf tensor
        Reconstructed x2
    """
    filters = d.get_shape().as_list()[-1]
    assert filters == s.get_shape().as_list()[-1]
    with tf.variable_scope('U') as scope:
        # Calculate x2 = s - Ud
        scope.reuse_variables()
        x2 = s - _residual_core(d, filters, 1, train, wd)

    with tf.variable_scope('P') as scope:
        # Calculate x_e = d + Px2
        scope.reuse_variables()
        x1 = d + _residual_core(x2, filters, 1, train, wd)

    return x1, x2


def complex_convolution(x, output_dim, size=3, stride=1, stddev=None,
                        wd=0.0, norm=1.0, name='conv2d', with_bias=False,
                        bias_start=0.0):
    """Function to do complex convolution

    In a similar way we have a convenience function, :py:func:`convolution` to
    wrap tf.nn.conv2d (create variables, add a relu, etc.), this function wraps
    :py:func:`cconv2d`. If you want more fine control over things, use
    cconv2d directly, but for most purposes, this function should do
    what you need. Adds the variables to tf.GraphKeys.REGULARIZATION_LOSSES
    if the wd parameter is positive.

    Parameters
    ----------
    x : :py:class:`tf.Tensor`
        The input variable
    output_dim : int
        number of filters to have
    size : int
        kernel spatial support
    stride : int
        what stride to use for convolution
    stddev : None or positive float
        Initialization stddev. If set to None, will use
        :py:func:`get_xavier_stddev`
    wd : None or positive float
        What weight decay to use
    norm : positive float
        Which regularizer to apply. E.g. norm=2 uses L2 regularization, and
        norm=p adds :math:`wd \\times ||w||_{p}^{p}` to the
        REGULARIZATION_LOSSES. See :py:func:`real_reg`.
    name : str
        The tensorflow variable scope to create the variables under
    with_bias : bool
        add a bias after convolution? (this will be ignored if batch norm is
        used)
    bias_start : complex float
        If a bias is used, what to initialize it to.

    Returns
    -------
    y : :py:class:`tf.Tensor`
        Result of applying complex convolution to x
    """

    varlist = []
    with tf.variable_scope(name):
        # Define the real and imaginary components of the weights
        w_shape = [size, size, x.get_shape().as_list()[-1], output_dim]
        w_r = variable_with_wd('w_real', w_shape, stddev, wd, norm)
        w_i = variable_with_wd('w_imag', w_shape, stddev, wd, norm)
        w = tf.complex(w_r, w_i)
        varlist.append(w)

        y = cconv2d(x, w, strides=[1, stride, stride, 1], name=name)
        y_r, y_i = tf.real(y), tf.imag(y)

        if with_bias:
            init = tf.constant_initializer(bias_start)
            b_r = tf.get_variable('b_real', [output_dim], initializer=init)
            b_i = tf.get_variable('b_imag', [output_dim], initializer=init)
            varlist.append(tf.complex(b_r, b_i))
            y_r = tf.add(y_r, b_r)
            y_i = tf.add(y_i, b_i)

    y = tf.complex(y_r, y_i)

    # Return the results
    return y


def complex_convolution_transpose(x, output_dim, shape, size=3, stride=1,
                                  stddev=None, wd=0.0, norm=1, name='conv2d'):
    """Function to do the conjugate transpose of complex convolution

    In a similar way we have a convenience function, :py:func:`convolution` to
    wrap tf.nn.conv2d (create variables, add a relu, etc.), this function wraps
    :py:func:`cconv2d_transpose`. If you want more fine control over things, use
    cconv2d_transpose directly, but for most purposes, this function should do
    what you need. Adds the variables to tf.GraphKeys.REGULARIZATION_LOSSES
    if the wd parameter is positive.

    We do not subtract the bias after doing the transpose convolution.

    Parameters
    ----------
    x : :py:class:`tf.Tensor`
        The input variable
    output_dim : int
        number of filters to have
    output_shape : list-like or 1-d Tensor
        list/tensor representing the output shape of the deconvolution op
    size : int
        kernel spatial support
    stride : int
        what stride to use for convolution
    stddev : None or positive float
        Initialization stddev. If set to None, will use
        :py:func:`get_xavier_stddev`
    wd : None or positive float
        What weight decay to use
    norm : positive float
        Which regularizer to apply. E.g. norm=2 uses L2 regularization, and
        norm=p adds :math:`wd \\times ||w||_{p}^{p}` to the
        REGULARIZATION_LOSSES. See :py:func:`real_reg`.
    name : str
        The tensorflow variable scope to create the variables under

    Returns
    -------
    y : :py:class:`tf.Tensor`
        Result of applying complex convolution transpose to x
    """

    varlist = []
    with tf.variable_scope(name):
        # Define the real and imaginary components of the weights
        w_shape = [size, size, x.get_shape().as_list()[-1], output_dim]
        w_r = variable_with_wd('w_real', w_shape, stddev, wd, norm)
        w_i = variable_with_wd('w_imag', w_shape, stddev, wd, norm)
        w = tf.complex(w_r, w_i)
        varlist.append(w)

        y = cconv2d_transpose(
            x, w, output_dim, strides=[1, stride, stride, 1], name=name)
        y_r, y_i = tf.real(y), tf.imag(y)

    y = tf.complex(y_r, y_i)

    # Return the results
    return y


def cconv2d(x, w, **kwargs):
    """ Performs convolution with complex inputs and weights

    Need to create the weights and feed to this function. If you want to have
    this done for you automatically, use :py:func:`complex_convolution`.

    Parameters
    ----------
    x : tf tensor
        input tensor
    w : tf tensor
        weights tensor
    kwargs : (key, val) pairs
        Same as tf.nn.conv2d

    Returns
    -------
    y : :py:class:`tf.Tensor`
        Result of applying convolution to x

    Notes
    -----
    Uses tf.nn.conv2d which I believe is actually cross-correlation.
    """
    default_args = {
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': "NHWC",
        'name': None
    }
    for key, val in kwargs.items():
        if key not in default_args.keys():
            raise KeyError(
                'Unknown argument {} for function tf.nn.conv2d'.format(key))
        else:
            default_args[key] = val

    x = tf.cast(x, tf.complex64)
    w = tf.cast(w, tf.complex64)
    x_r = tf.real(x)
    x_i = tf.imag(x)
    w_r = tf.real(w)
    w_i = tf.imag(w)
    conv = lambda x, w: tf.nn.conv2d(x, w, **default_args)
    y_r = conv(x_r, w_r) - conv(x_i, w_i)
    y_i = conv(x_i, w_r) + conv(x_r, w_i)

    return tf.complex(y_r, y_i)


def cconv2d_transpose(y, w, output_shape, **kwargs):
    """ Performs transpose convolution with complex outputs and weights.

    Need to create the weights and feed to this function. If you want to have
    this done for you automatically, use
    :py:func:`complex_convolution_transpose`.

    Parameters
    ----------
    x : tf tensor
        input tensor
    w : tf tensor
        weights tensor
    kwargs : (key, val) pairs
        Same as tf.nn.conv2d_transpose

    Notes
    -----
    Takes the complex conjugate of w before doing convolution. Uses
    tf.nn.conv2d_transpose which I believe is actually convolution.

    Returns
    -------
    y : :py:class:`tf.Tensor`
        Result of applying convolution to x
    """
    default_args = {
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': "NHWC",
        'name': None
    }
    for key, val in kwargs.items():
        if key not in default_args.keys():
            raise KeyError(
                'Unknown argument {} for function '.format(key) +
                'tf.nn.conv2d_transpose')
        else:
            default_args[key] = val

    y = tf.cast(y, tf.complex64)
    w = tf.cast(w, tf.complex64)
    y_r = tf.real(y)
    y_i = tf.imag(y)
    w_r = tf.real(w)
    w_i = -tf.imag(w)
    conv = lambda y, w: tf.nn.conv2d_transpose(
        y, w, output_shape, **default_args)
    x_r = conv(y_r, w_r) - conv(y_i, w_i)
    x_i = conv(y_i, w_r) + conv(y_r, w_i)
    x_r = tf.reshape(x_r, output_shape)
    x_i = tf.reshape(x_i, output_shape)

    return tf.complex(x_r, x_i)


def separable_conv_with_pad(x, h_row, h_col, stride=1):
    """ Function to do spatial separable convolution.

    The filter weights must already be defined. It will use symmetric extension
    before convolution.

    Parameters
    ----------
    x : :py:class:`tf.Tensor` of shape [Batch, height, width, c]
        The input variable. Should be of shape
    h_row : tf tensor of shape [1, l, c_in, c_out]
        The spatial row filter
    h_col : tf tensor of shape [l, 1, c_in, c_out]
        The column filter.
    stride : int
        What stride to use on the convolution.

    Returns
    -------
    y : :py:class:`tf.Tensor`
        Result of applying convolution to x
    """
    # Do the row filter first:
    if tf.is_numeric_tensor(h_row):
        h_size = h_row.get_shape().as_list()
    else:
        h_size = h_row.shape

    assert h_size[0] == 1
    pad = h_size[1] // 2
    if h_size[1] % 2 == 0:
        y = tf.pad(x, [[0, 0], [0, 0], [pad - 1, pad], [0, 0]], 'SYMMETRIC')
    else:
        y = tf.pad(x, [[0, 0], [0, 0], [pad, pad], [0, 0]], 'SYMMETRIC')
    y = tf.nn.conv2d(y, h_row, strides=[1, stride, stride, 1],
                     padding='VALID')

    # Now do the column filtering
    if tf.is_numeric_tensor(h_col):
        h_size = h_col.get_shape().as_list()
    else:
        h_size = h_col.shape

    assert h_size[1] == 1
    pad = h_size[0] // 2
    if h_size[0] % 2 == 0:
        y = tf.pad(y, [[0, 0], [pad - 1, pad], [0, 0], [0, 0]], 'SYMMETRIC')
    else:
        y = tf.pad(y, [[0, 0], [pad, pad], [0, 0], [0, 0]], 'SYMMETRIC')
    y = tf.nn.conv2d(y, h_col, strides=[1, stride, stride, 1],
                     padding='VALID')

    assert x.get_shape().as_list()[1:3] == y.get_shape().as_list()[1:3]

    return y


def _get_var_name(x):
    """ Find the name of the variable by stripping off the scopes

    Notes
    -----
    A typical name will be scope1/scope2/.../name/kernel:0.
    This function serves to split off the scopes and return kernel
    """
    split_colon = x.name.split(':')[0]
    slash_strs = split_colon.split('/')
    #  last_two = slash_strs[-2] + '/' + slash_strs[-1]
    last_one = slash_strs[-1]
    return last_one


def get_static_shape_dyn_batch(x):
    """Returns a tensor representing the static shape of x but keeping the batch
    unkown"""
    batch = tf.shape(x)[0]
    static = x.get_shape()
    return tf.concat([[batch], static[1:]], axis=0)


def get_xavier_stddev(shape, uniform=False, factor=1.0, mode='FAN_AVG'):
    """Get the correct stddev for a set of weights

    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. This initializer use the following formula:

    .. code:: python

      if mode='FAN_IN': # Count only number of input connections.
          n = fan_in
      elif mode='FAN_OUT': # Count only number of output connections.
          n = fan_out
      elif mode='FAN_AVG': # Average number of inputs and output connections.
          n = (fan_in + fan_out)/2.0
          truncated_normal(shape, 0.0, stddev=sqrt(factor/n))

    * To get `Delving Deep into Rectifiers`__, use::

          factor=2.0
          mode='FAN_IN'
          uniform=False

      __ http://arxiv.org/pdf/1502.01852v1.pdf

    * To get `Convolutional Architecture for Fast Feature Embedding`__ , use::

          factor=1.0
          mode='FAN_IN'
          uniform=True

      __ http://arxiv.org/abs/1408.5093

    * To get `Understanding the difficulty of training deep feedforward neural
      networks`__ use::

          factor=1.0
          mode='FAN_AVG'
          uniform=True

      __ http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

    * To get `xavier_initializer` use either::

        factor=1.0
        mode='FAN_AVG'
        uniform=True

      or::

        factor=1.0
        mode='FAN_AVG'
        uniform=False

    Parameters
    ----------
    factor: float
        A multiplicative factor.
    mode : str
        'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
    uniform : bool
        Whether to use uniform or normal distributed random initialization.
    seed : int
        Used to create random seeds. See `tf.set_random_seed`__
        for behaviour.

        __ https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    dtype : tf.dtype
        The data type. Only floating point types are supported.

    Returns
    -------
    out : float
        The stddev/limit to use that generates tensors with unit variance.

    Raises
    ------
    ValueError : if `dtype` is not a floating point type.
    TypeError : if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].
    """
    if shape:
        fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        fan_out = float(shape[-1])
    else:
        fan_in = 1.0
        fan_out = 1.0

    for dim in shape[:-2]:
        fan_in *= float(dim)
        fan_out *= float(dim)

    if mode == 'FAN_IN':
        # Count only number of input connections.
        n = fan_in
    elif mode == 'FAN_OUT':
        # Count only number of output connections.
        n = fan_out
    elif mode == 'FAN_AVG':
        # Average number of inputs and output connections.
        n = (fan_in + fan_out) / 2.0

    if uniform:
        # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
        limit = math.sqrt(3.0 * factor / n)
        return limit
        #  return random_ops.random_uniform(shape, -limit, limit,
                                         #  dtype, seed=seed)
    else:
        # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
        trunc_stddev = math.sqrt(1.3 * factor / n)
        return trunc_stddev
        #  return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                             #  seed=seed)


def real_reg(w, wd=0.01, norm=2):
    """ Apply regularization on real weights

    norm can be any positive float. Of course the most commonly used values
    would be 2 and 1 (for L2 and L1 regularization), but you can experiment by
    making it some value in between. A value of p returns:

    .. math::

        wd \\times \\sum_{i} ||w_{i}||_{p}^{p}

    Parameters
    ----------
    w : :py:class:`tf.Tensor`
        The weights to regularize
    wd : positive float, optional (default=0.01)
        Regularization parameter
    norm : positive float, optional (default=2)
        The norm to use for regularization. E.g. set norm=1 for the L1 norm.

    Returns
    -------
    reg_loss : :py:class:`tf.Tensor`
        The loss. This method does not add anything to the REGULARIZATION_LOSSES
        collection. The calling function needs to do that.

    Raises
    ------
    ValueError : If norm is less than 0
    """
    if wd is None or wd == 0 or norm is None:
        return
    if norm <= 0:
        raise ValueError('Can only take positive norms, not {}'.format(norm))

    if norm == 2:
        # L2 Loss computes half of the sum of squares
        reg_loss = tf.nn.l2_loss(w)
    elif norm == 1:
        mag = tf.abs(w)
        reg_loss = tf.reduce_sum(mag)
    else:
        mag = tf.abs(w)
        reg_loss = (1/norm) * tf.reduce_sum(mag**norm)

    reg_loss = tf.multiply(reg_loss, wd, name='weight_loss')
    return reg_loss
    #  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg_loss)


def complex_reg(w, wd=0.01, norm=1):
    """ Apply regularization on complex weights.

    norm can be any positive float. Of course the most commonly used values
    would be 2 and 1 (for L2 and L1 regularization), but you can experiment by
    making it some value in between. A value of p returns:

    .. math::

        wd \\times \\sum_{i} ||w_{i}||_{p}^{p}


    Parameters
    ----------
    w : :py:class:`tf.Tensor` (dtype=complex)
        The weights to regularize
    wd : positive float, optional (default=0.01)
        Regularization parameter
    norm : positive float, optional (default=1)
        The norm to use for regularization. E.g. set norm=1 for the L1 norm.

    Returns
    -------
    reg_loss : :py:class:`tf.Tensor`
        The loss. This method does not add anything to the REGULARIZATION_LOSSES
        collection. The calling function needs to do that.

    Raises
    ------
    ValueError : If norm is less than 0

    Notes
    -----
    Can call this function with real weights too, making it perhaps a better
    de-facto function to call, as it able to handle both cases.
    """
    if wd is None or wd == 0 or norm is None:
        return
    if norm <= 0:
        raise ValueError('Can only take positive norms, not {}'.format(norm))

    # Check the weights input. Use the real regularizer if weights are purely
    # real
    if w.dtype.is_floating:
        return real_reg(w, wd, norm)

    # L2 is a special regularization where we can regularize the real and
    # imaginary components independently. All other types we need to combine
    # them to get the magnitude.
    if norm == 2:
        # L2 Loss computes half of the sum of squares
        reg_loss = tf.nn.l2_loss(tf.real(w)) + tf.nn.l2_loss(tf.imag(w))
    elif norm == 1:
        mag = tf.abs(w)
        reg_loss = tf.reduce_sum(mag)
    else:
        mag = tf.abs(w)
        reg_loss = (1/norm) * tf.reduce_sum(mag**norm)

    reg_loss = tf.multiply(reg_loss, wd, name='weight_loss')
    return reg_loss
    #  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg_loss)
