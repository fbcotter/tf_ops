from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import logging

import dtcwt
from dtcwt.coeffs import biort as _biort
from dtcwt.tf import Transform2d, Pyramid


def _activation_summary(x, name):
    """ Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Parameters
    ----------
    x: tf tensor
        tensor for which you want to create summaries

    """
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))


def _dtcwt_correct_phases(X, inv=False):
    """ Corrects wavelet phases so the centres line up.

    i.e. This makes sure the centre of the real wavelet is a zero crossing for
    all orientations.

    For the inverse path, we needed to multiply the coefficients by the phase
    correcting factor of: [1j, -1j, 1j, -1, 1, -1].
    For the forward path, we divide by these numbers, or multiply by their
    complex conjugate.

    Parameters
    ----------
    X: ndarray of complex tensorflow floats of shape (batch, ..., 6)
        DTCWT input. Should have final dimension of 6
    inv : bool
        Whether this is the forward or backward pass.  Default is false (i.e.
        forward)
    """

    with tf.variable_scope('correct_phases'):
        w_fwd = tf.constant([1j, -1j, 1j, -1, 1, -1], tf.complex64)
        w_inv = tf.constant([-1j, 1j, -1j, -1, 1, -1], tf.complex64)
        w = w_inv if inv else w_fwd

        f = lambda x: x * w

        # Check if the input is an iterable or a single array
        if type(X) is list or type(X) is tuple:
            Y = list()
            for x in X:
                Y.append(f(x))
            Y = tuple(Y)
        else:
            Y = f(X)

    return Y


def lazy_wavelet(x):
    """ Performs a lazy wavelet split on a tensor.

    Designed to work nicely with the lifting blocks.

    Output will have 4 times as many channels as the input, but will have one
    quarter the spatial size.  I.e. if x is a tensor of size (batch, h, w, c),
    then the output will be of size (batch, h/2, w/2 4*c). The first c channels
    will be the A samples::

        Input image

        A B A B A B ...
        C D C D C D ...
        A B A B A B ...
        ...

    Then the next c channels will be from the B channels, and so on.

    Notes
    -----
    If the spatial size is not even, then will mirror to make it even before
    downsampling.

    Parameters
    ----------
    x : tf tensor
        Input to apply lazy wavelet transform to.

    Returns
    -------
    y : tf tensor
        Result after applying transform.
    """
    xshape = x.get_shape().as_list()
    pad = [[0,0], [0,0], [0,0], [0,0]]
    if xshape[1] % 2 == 1:
        pad[1][1] = 1
    if xshape[2] % 2 == 1:
        pad[2][1] = 1
    x = tf.pad(x, pad, 'SYMMETRIC')
    A, B = x[:, ::2, ::2], x[:, ::2, 1::2]
    C, D = x[:, 1::2, ::2], x[:, 1::2, 1::2]
    y = tf.concat([A, B, C, D], axis=-1)
    return y


def lazy_wavelet_inv(x, out_size=None):
    """ Performs the inverse of a lazy wavelet transform - a 'lazy recombine'

    Designed to work nicely with the lifting blocks.

    Output will have 1/4 as many channels as the input, but will have one
    quadruple the spatial size.  I.e. if x is a tensor of size (batch, h, w, c),
    then the output will be of size (batch, 2*h, 2*w c/4). If we call the first
    c channels the A group, then the second c the B group, and so on, the output
    image will be interleaved like so::

        Output image

        A B A B A B ...
        C D C D C D ...
        A B A B A B ...
        ...

    Notes
    -----
    If the forward lazy wavelet needed padding, then we should be undoing it
    here. For this, specify the out_size of the resulting tensor.

    Parameters
    ----------
    x : tf tensor
        Input to apply lazy wavelet transform to.
    out_size : tuple of 4 ints or None
        What the output size should be of the resulting tensor. The batch size
        will be ignored, but the spatial and channel size need to be correct.
        For an input spatial size of (h, r), the spatial dimensions (the 2nd and
        3rd numbers in the tuple) should be either (2*h, 2*r),
        (2*h-1, 2*r), (2*h, 2*r-1) or (2*h-1, 2*r-1). Will raise a ValueError if
        not one of these options. Can also be None, in which (2*h, 2*r) is
        assumed. The channel size should be 1/4 of the input channel size.

    Returns
    -------
    y : tf tensor
        Result after applying transform.

    Raises
    ------
    ValueError when the out_size is invalid, or if the input tensor's channel
    dimension is not divisible by 4.
    """
    xshape = x.get_shape().as_list()
    # Check the channel axis
    if xshape[-1] % 4 != 0:
        raise ValueError('Input tensor needs to have 4k channels')
    if xshape[-1] // 4 != out_size[-1]:
        raise ValueError('Out tensor needs to have 1/4 channels of input')
    k = xshape[-1] // 4

    # Check the spatial axes
    dbl_size = [2*xshape[1], 2*xshape[2]]
    if out_size is None:
        out_size = dbl_size
    else:
        out_size = out_size[1:3]
    if out_size[0] != dbl_size[0] and out_size[0] != dbl_size[0] - 1:
        raise ValueError('Row size in out_size incorrect')
    if out_size[1] != dbl_size[1] and out_size[1] != dbl_size[1] - 1:
        raise ValueError('Col size in out_size incorrect')

    A, B = x[:, :, :, :k], x[:, :, :, k:2*k]
    C, D = x[:, :, :, 2*k:3*k], x[:, :, :, 3*k:]

    # Create the even and odd rows by careful stacking and reshaping.
    r_o = tf.stack([A, B], axis=3)
    r_o = tf.reshape(r_o, [-1, xshape[1], xshape[2]*2, k])
    r_e = tf.stack([C, D], axis=3)
    r_e = tf.reshape(r_e, [-1, xshape[1], xshape[2]*2, k])
    y = tf.stack([r_o, r_e], axis=2)
    y = tf.reshape(y, [-1, xshape[1]*2, xshape[2]*2, k])

    # Cut off the edges if need be.
    y = y[:, :out_size[0], :out_size[1], :]
    return y


def phase(z):
    """Calculate the elementwise arctan of z, choosing the quadrant correctly.

    Quadrant I: arctan(y/x)
    Qaudrant II: π + arctan(y/x) (phase of x<0, y=0 is π)
    Quadrant III: -π + arctan(y/x)
    Quadrant IV: arctan(y/x)

    Parameters
    ----------
    z : tf.complex64 datatype of any shape

    Returns
    -------
    y : tf.float32
        Angle of z
    """
    if z.dtype == tf.complex128:
        dtype = tf.float64
    else:
        dtype = tf.float32
    x = tf.real(z)
    y = tf.imag(z)
    xneg = tf.cast(x < 0.0, dtype)
    yneg = tf.cast(y < 0.0, dtype)
    ypos = tf.cast(y >= 0.0, dtype)

    offset = xneg * (ypos - yneg) * np.pi

    return tf.atan(y / x, name='phase') + offset


def filter_across(X, H, inv=False):
    """ Do 1x1 convolution as a matrix product.

    Parameters
    ----------
    X : tf tensor of shape (batch, ..., 12)
        The input. Must have final dimension 12 - corresponding to the 6
        orientations of the DTCWT and their conjugates.
    H : tf tensor of shape (12, 12)
        The filter.
    inv : bool
        True if this is the inv operation. If inverse, we first take the
        transpose conjugate of H. This way you can call the filter_across
        function with the same H for the fwd and inverse. Default is False.

    Returns
    -------
        y = XH
    """
    # Assert shape of input and filter are ok
    assert X.get_shape().as_list()[-1] == 12
    assert H.get_shape().as_list() == [12, 12]

    if inv:
        H = tf.conj(tf.transpose(H, [1, 0]))

    # H is a block circulant matrix, that is there to represent the convolution
    # over the orientations of X with our underlying filter f. As it is the
    # convolution of 2 complex signals, we do not want to take the complex
    # conjugate of either of them, hence we use tf.tensordot
    y = tf.tensordot(X, tf.transpose(H, [1, 0]), axes=[[-1], [0]])
    return y


def up_with_zeros(x, y):
    """Upsample tensor by inserting zeros at new positions.

    This keeps the input sparse in the new domain.
    For the moment, only works on square inputs.

    Parameters
    ----------
    x : tf tensor
        The input
    y : int
        The upsample rate (can only do 2 or 4 currently)
    """
    assert (y == 2 or y == 4)
    xshape = x.get_shape().as_list()
    assert (len(xshape) == 4)
    assert (xshape[1] == xshape[2])

    newshape = y * xshape[1]
    a = tf.ones((xshape[1], 1), tf.float32)
    b = tf.zeros((xshape[1], 1), tf.float32)
    if (y == 2):
        c = tf.reshape(tf.stack([a, b], axis=1), [1, newshape])
    elif (y == 4):
        c = tf.reshape(tf.stack([b, a, b, b], axis=1), [1, newshape])
    d = tf.transpose(c, [1, 0])

    # Create a [newshape, newshape] matrix by taking the matrix product of
    # d and c. This will be a matrix that is zeros everywhere, but will have
    # xshape*xshape 1s, evenly spaced through it.
    coeff_matrix = tf.matmul(d, c)
    coeff_matrix = tf.expand_dims(coeff_matrix, axis=0)
    coeff_matrix = tf.expand_dims(coeff_matrix, axis=-1)
    assert coeff_matrix.get_shape().as_list() == [1, newshape, newshape, 1]

    # Upsample using nearest neighbour
    X = tf.image.resize_nearest_neighbor(x, [newshape, newshape])

    # Mask using the above coeff_matrix
    X = X * coeff_matrix

    return X


def add_conjugates(X):
    """ Concatenate tensor with its conjugates.

    The DTCWT returns an array of 6 orientations for each coordinate,
    corresponding to the 6 orientations of [15, 45, 75, 105, 135, 165]. We can
    get another 6 rotations by taking complex conjugates of these.

    Parameters
    ----------
    X : tf tensor of shape (batch, ..., 6)

    Returns
    -------
    Y : tf tensor of shape (batch, .... 12)
    """
    f = lambda x: tf.concat([x, tf.conj(x)], axis=-1)
    if type(X) is list or type(X) is tuple:
        Y = list()
        for x in X:
           Y.append(f(x))
    else:
        Y = f(X)
    return Y


def collapse_conjugates(X):
    """ Invert the add_conjugates function. I.e. go from 12 dims to 6

    To collapse, we add the first 6 dimensions with the conjugate of the last 6
    and then divide by 2.

    Parameters
    ----------
    X : tf tensor of shape (batch, ..., 12)

    Returns
    -------
    Y : tf tensor of shape (batch, .... 6)
    """
    ax = lambda x: len(x.get_shape().as_list()) - 1
    split = lambda x: tf.split(x, [6,6], axis=ax(x))
    def f(x):
        t1, t2 = split(x)
        return 0.5 * (t1 + tf.conj(t2))
    if type(X) is list or type(X) is tuple:
        Y = list()
        for x in X:
            Y.append(f(x))
    else:
        Y = f(X)
    return Y


def response_normalization(x, power=2):
    """ Function to spread out the activations.

    The aim is to keep the top activation as it is, and send the others towards
    zero. We can do this by mapping our data through a polynomial function,
    ax^power. We adjust a so that the max value remains unchanged.

    Negative inputs should not happen as we are competing after a magnitude
    operation. However, they can sometimes occur due to upsampling that may
    happen. If this is the case, we clip them to 0.
    """
    m = tf.expand_dims(tf.reduce_max(x, axis=-1), axis=-1)
    # Clip negative values
    x = tf.maximum(x, 0.0)
    #  factor = tf.exp(power * m)
    #  return tf.exp(power * x) * x/factor - 1
    a = 1 / m**(power - 1)
    return x**power / a


def wavelet(x, nlevels, biort='near_sym_b_bp', qshift='qshift_b_bp',
            data_format="nhwc"):
    """ Perform an nlevel dtcwt on the input data.

    Parameters
    ----------
    x: tf tensor of shape (batch, h, w) or (batch, h, w, c)
        The input to be transformed. If the input has a channel dimension, the
        dtcwt will be applied to each of the channels independently.
    nlevels : int
        the number of scales to use. 0 is a special case, if nlevels=0, then we
        return a lowpassed version of the x, and Yh and Yscale will be
        empty lists
    biort : str
        which biorthogonal filters to use. 'near_sym_b_bp' are my favourite, as
        they have 45° and 135° filters with the same period as the others.
    qshift : str
        which quarter shift filters to use. These should match up with the
        biorthogonal used. 'qshift_b_bp' are my favourite for the same reason.
    data_format : str
        An optional string of the form "nchw" or "nhwc" (for 4D data), or "nhw"
        or "hwn" (for 3D data). This specifies the data format of the input.
        E.g. If format is "nchw" (the default), then data is in the form [batch,
        channels, h, w]. If the format is "nhwc", then the data is in the form
        [batch, h, w, c].

    Returns
    -------
    out : a tuple of (lowpass, highpasses and scales).

        * Lowpass is a tensor of the lowpass data. This is a real float.  If
          x has shape [batch, height, width, channels], the dtcwt will be
          applied independently for each channel and combined.
        * Highpasses is a list of length <nlevels>, each entry has the six
          orientations of wavelet coefficients for the given scale. These are
          returned as tf.complex64 data type.
        * Scales is a list of length <nlevels>, each entry containing the
          lowpass signal that gets passed to the next level of the dtcwt
          transform.
    """

    # If nlevels was 0, lowpass the input
    with tf.variable_scope('wavelet'):
        if nlevels == 0:
            Yh, Yscale = [], []
            filters = _biort(biort)
            h0o = np.reshape(filters[0], [1, -1, 1, 1])
            h0oT = np.transpose(h0o, [1, 0, 2, 3])
            Xshape = x.get_shape().as_list()

            # Put the channels to the batch dimension and back after
            X = tf.reshape(x, [-1, Xshape[1], Xshape[2], 1])
            Yl = separable_conv_with_pad(X, h0o, h0oT)
            Yl = tf.reshape(Yl, [-1, Xshape[1], Xshape[2], Xshape[3]])
        else:
            transform = Transform2d(biort=biort, qshift=qshift)
            # Use either forward_channels or forward, depending on the input
            # shape
            noch = len(['batch', 'height', 'width'])
            ch = len(['batch', 'height', 'width', 'channel'])
            l = len(x.get_shape().as_list())
            if l == noch:
                data_format = 'nhw'
                Yl, Yh, Yscale = dtcwt.utils.unpack(
                    transform.forward_channels(
                        x, data_format, nlevels=nlevels, include_scale=True),
                    'tf')

            elif l == ch:
                Yl, Yh, Yscale = dtcwt.utils.unpack(
                    transform.forward_channels(
                        x, data_format, nlevels=nlevels, include_scale=True),
                    'tf')
            else:
                raise ValueError("Unkown length {} for wavelet block".format(l))

    return Yl, _dtcwt_correct_phases(Yh), Yscale


def wavelet_inv(Yl, Yh, biort='near_sym_b_bp', qshift='qshift_b_bp',
                data_format="nhwc"):
    """ Perform an nlevel inverse dtcwt on the input data.

    Parameters
    ----------
    Yl : :py:class:`tf.Tensor`
        Real tensor of shape (batch, h, w) or (batch, h, w, c) holding the
        lowpass input. If the shape has a channel dimension, then c inverse
        dtcwt's will be performed (the other inputs need to also match this
        shape).
    Yh : list(:py:class:`tf.Tensor`)
        A list of length nlevels. Each entry has the high pass for the scales.
        Shape has to match Yl, with a 6 on the end.
    biort : str
        Which biorthogonal filters to use. 'near_sym_b_bp' are my favourite, as
        they have 45° and 135° filters with the same period as the others.
    qshift : str
        Which quarter shift filters to use. These should match up with the
        biorthogonal used. 'qshift_b_bp' are my favourite for the same reason.
    data_format : str
        An optional string of the form "nchw" or "nhwc" (for 4D data), or "nhw"
        or "hwn" (for 3D data). This specifies the data format of the input.
        E.g. If format is "nchw" (the default), then data is in the form [batch,
        channels, h, w]. If the format is "nhwc", then the data is in the form
        [batch, h, w, c].

    Returns
    -------
    X : :py:class:`tf.Tensor`
        An input of size [batch, h', w'], where h' and w' will be larger than
        h and w by a factor of 2**nlevels
    """

    with tf.variable_scope('wavelet_inv'):
        Yh = _dtcwt_correct_phases(Yh, inv=True)
        transform = Transform2d(biort=biort, qshift=qshift)
        pyramid = Pyramid(Yl, Yh)
        X = transform.inverse_channels(pyramid, data_format=data_format)

    return X


def combine_channels(x, dim, combine_weights=None):
    """ Sum over over the specified dimension with optional summing weights.

    Parameters
    ----------
    x : tf tensor
        Tensor which will be summed over
    dim : int
        which dimension to sum over
    combine_weights : None or list of floats
        The weights to use when summing. If left as none, the weights will be 1.

    Returns
    -------
    Y : :py:class:`tf.Tensor` of shape one less than x
    """
    if combine_weights is not None:
        # reshape the weights to make them match the specified dimension
        s = x.get_shape().as_list()
        l = [-1] + [1 for x in range(dim, len(s) - 1)]
        w = tf.reshape(tf.constant(combine_weights, tf.complex64), l)
        Y = tf.multiply(x, w, name='weighted_combination')

    return tf.reduce_sum(Y, axis=3)


def complex_mag(x, bias_start=0.0, learnable_bias=False,
                combine_channels=False, combine_weights=None,
                return_direction=False):
    """ Perform wavelet magnitude operation on complex highpass outputs.

    Will subtract a bias term from the sum of the real and imaginary parts
    squared, before taking the square root. I.e. y=√max(ℜ²+ℑ²-b², 0). This bias
    can be learnable or set.

    Parameters
    ----------
    x : :py:class:`tf.Tensor`
        Tf tensor of shape (batch, h, w, ..., 6)
    bias_start : float
        What the b term will be set to to begin with.
    learnable_bias : bool
        If true, bias will be a tensorflow variable, and will be added to the
        trainable variables list. Bias will have shape: [channels, 6] if the
        input had channels, or simply [6] if it did not.
    combine_channels : bool
        Whether to combine the channels magnitude or not.  If true, the output
        will be [batch, height, width, 6]. Combination will be done by simply
        summing up the square roots. I.e.::

            √(ℜ₁²+ℑ₁²+ℜ₂²+ℑ₂²+...+ℜc²+ℑc² - b²)

        In this situation, the bias will have shape [6]
    combine_weights : bool
        A list of weights used to combine channels. This must be of the same
        length as channels. If omitted, the channels will be combined equally.
    return_direction: bool
        If true, also return a unit magnitude direction vector

    Returns
    -------
    out : tuple of (abs(x),) or (abs(x), unit(x))
        Tensor of same shape as input (unless combine_channels was True),
        which is the magnitude of the real and imaginary components. Will
        return a tuple of (abs(in), angle(in)) if return_phase is True)
    """
    assert x.dtype == tf.complex64

    # Find the shape of the input (may or may not have channels)
    in_shape = x.get_shape().as_list()
    ch = len(['batch', 'height', 'width', 'channel', 'angles'])
    noch = len(['batch', 'height', 'width', 'angles'])
    if len(in_shape) == ch:
        has_channels = True
        channel_ax = 3
        angle_ax = 4
    elif len(in_shape) == noch:
        angle_ax = 3
        has_channels = False
        if combine_channels:
            logging.warn(
                'Cannot combine channels for tensor of shape {}'.format(
                    in_shape) + '. Continuing as if it were set to false')
            combine_channels = False
    else:
        raise ValueError('Unable to handle input shape to complex_mag function')

    with tf.variable_scope('mag'):
        if combine_channels:
            b_shape = [in_shape[angle_ax]]
        else:
            if has_channels:
                b_shape = [in_shape[channel_ax], in_shape[angle_ax]]
            else:
                b_shape = [in_shape[angle_ax]]

        if learnable_bias:
            b = tf.get_variable(
                'b',
                b_shape,
                initializer=tf.constant_initializer(bias_start))
            _activation_summary(b, 'biases')
        else:
            b = bias_start

        #  Combine the magnitudes of the channels
        if combine_channels:
            # Instead of combining the channels equally, allow the option to
            # combine them in a weighted sense. E.g. if the weights were the
            # RGB channels, we would want to put more weight on the G channel.
            if combine_weights is not None:
                w = tf.reshape(tf.constant(combine_weights, tf.complex64),
                               [-1, 1])
                x = tf.multiply(x, w, name='weighted_combination')

            fx = tf.reduce_sum(
                tf.real(x)**2 + tf.imag(x)**2,
                axis=channel_ax)

            # Add the squared magnitude for each channel then subtract the bias
            mag2 = tf.maximum(0.0, fx - b**2)
            m = tf.sqrt(mag2, name='mag')
            if return_direction:
                m_big = tf.stack([m] * in_shape[channel_ax], axis=-2)
                p_r = tf.real(x) / m_big
                p_i = tf.imag(x) / m_big
                # Handle any nans that could have popped up from 0 mag
                p = tf.where(tf.equal(m_big, 0.0),
                             tf.zeros_like(x),
                             tf.complex(p_r, p_i))

        else:
            # Add the squared magnitude and subtract the bias
            mag2 = tf.maximum(0.0, tf.real(x)**2 + tf.imag(x)**2 -
                              b**2)
            m = tf.sqrt(mag2, name='mag')
            if return_direction:
                p_r = tf.real(x) / m
                p_i = tf.imag(x) / m
                # Handle any nans that could have popped up from 0 mag
                p = tf.where(tf.equal(m, 0.0),
                             tf.zeros_like(x),
                             tf.complex(p_r, p_i))

    if return_direction:
        return m, p
    else:
        return m


def separable_conv_with_pad(x, h_row, h_col, stride=1):
    """ Function to do spatial separable convolution.

    The filter weights must already be defined. It will use symmetric extension
    before convolution.

    Parameters
    ----------
    x : tf variable of shape [Batch, height, width, c]
        The input variable. Should be of shape
    h_row : tf tensor of shape [1, l, c_in, c_out]
        The spatial row filter
    h_col : tf tensor of shape [l, 1, c_in, c_out]
        The column filter.
    stride : int
        What stride to use on the convolution.

    Returns
    -------
    y : tf variable
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
