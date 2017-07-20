Fergal's TF Ops
===============
This library provides some convenience functions for doing some common
operations in tensorflow. I recommend you also look at the tf.layers module, as
there is a lot of overlap; I use these functions as some of those in tf.layers
are a little bit unclear about how they work.

In addition, I define operations to do complex valued convolution (complex 
inputs and complex weights).

If you are using tensorflow on a shared GPU server and want to control how many
GPUs it grabs, have a look `py3nvml <https://github.com/fbcotter/py3nvml.git>`_,
in particular the ``py3nvml.grab_gpus()`` function.

.. _installation-label:

Installation
------------
Direct install from github (useful if you use pip freeze). To get the master
branch, try::

    $ pip install -e git+https://github.com/fbcotter/tf_ops#egg=tf_ops

or for a specific tag (e.g. 0.0.1), try::

    $ pip install -e git+https://github.com/fbcotter/tf_ops.git@0.0.1#egg=tf_ops

Download and pip install from Git::

    $ git clone https://github.com/fbcotter/tf_ops
    $ cd tf_ops
    $ pip install -r requirements.txt
    $ pip install -e .

I would recommend to download and install (with the editable flag), as it is
likely you'll want to tweak things/add functions more quickly than I can handle
pull requests.

Further documentation
---------------------

There is `more documentation <http://tf-ops.readthedocs.io>`_
available online and you can build your own copy via the Sphinx documentation
system::

    $ python setup.py build_sphinx

Compiled documentation may be found in ``build/docs/html/`` (index.html will be
the homepage)
