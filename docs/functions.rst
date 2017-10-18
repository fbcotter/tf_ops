Function List
=============

Layer Functions
---------------

The following functions are high level convenience functions. They will create
weights, do the intended layer, and can add regularizations, non-linearities,
batch-norm and other helpful features.

.. automodule:: tf_ops.general
    :members: residual, lift_residual, lift_residual_inv, complex_convolution, 
      complex_convolution_transpose
    :show-inheritance:

Initializers and Regularizers
-----------------------------

The following functions are helpers to initialize weights and add regularizers
to them.

.. automodule:: tf_ops.general
    :members: variable_with_wd, get_xavier_stddev, real_reg, complex_reg
    :show-inheritance:

Losses and Summaries
--------------------

.. automodule:: tf_ops.general
    :members: loss, variable_summaries
    :show-inheritance:

Core Functions
--------------

Some new functions to do things tensorflow currently doesn't do.

.. automodule:: tf_ops.general
    :members: cconv2d, cconv2d_transpose, separable_conv_with_pad
    :show-inheritance:

Wavelet Functions
-----------------

On top of the above general functions, there are also some DTCWT based wavelet
functions.  

.. automodule:: tf_ops.wave_ops
    :members:
    :show-inheritance:
