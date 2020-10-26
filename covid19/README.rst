Bayesian Inference of Epidemic Models
---------------------------------------

coming soon...


Project structure
=================

- ``epidemics`` - The Python module and files.
- ``epidemics/data`` - Python files for fetch and preprocessing data; data files.
- ``epidemics/data/files`` - Data files.
- ``epidemics/data/downloads`` - Raw files downloaded by the Python scripts.
- ``epidemics/data/cache`` - Files processed by the Python scripts.
- ``src/epidemics`` - The C++ code.
- ``src/epidemics/model/country`` - C++ country-level model implementations.
- ``src/epidemics/model/cantons`` - C++ canton-level model implementations.
- ``test/py`` - Python unit tests.
- ``test/cpp`` - C++ unit tests.


Compilation
===========

Install the boost library by following
`these instructions <https://www.boost.org/doc/libs/1_66_0/more/getting_started/unix-variants.html>`_. In macOS you can run

.. code-block:: bash

  brew install boost



From the repository root folder do:

.. code-block:: bash

    pip3 install jinja2
    git submodule update --init --recursive
    mkdir -p build
    cd build
    cmake ..
    make


Tests
=====

To run the tests, run the following command (from the repository root):

.. code-block:: bash

    cd tests
    ./run_all_tests.sh

To run only Python tests, run ``cd tests/py && ./run.sh``.
To run only C++ tests, run ``cd build && ./libepidemics_unittests``.


Code formatting
===============

Python
~~~~~~

Install `yapf <https://github.com/google/yapf>`_.

.. code-block::

    pip3 install yapf

Format a file in-place

.. code-block::

    yapf -i FILE

Format all python files recursively in a directory in-place

.. code-block::

    yapf -ir DIR

C++
~~~

Install ``clang-format`` as part of `clang` using your package manager
or download a
`static build <http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-sles11.3.tar.xz>`_

Format a file in-place

.. code-block::

    clang-format -i FILE


Troubleshooting
===============

If the changes in the code are not reflected in the results, try erasing ``epidemics/data/cache`` and ``epidemica/data/downloads`` folders.
The cache decorators attempt to detect changes in the code, but may not always be successful.


Adding a new country-level model (C++)
======================================

Follow these steps to create a new C++ country-level model. The steps are shown on an example of creating a XYZ model from the existing SIR model.

1. Make a copy of ``src/epidemics/models/country/sir.h`` and name it ``xyz.h``.

2. Change the ``sir`` namespace to ``xyz``.

2. Update the ``Parameters`` struct.

3. Update the ``State`` struct: change the number of states in ``StateBase`` parent class, and customize named getters.

4. Update the model, the ``Solver::rhs`` function.

5. Edit ``src/epidemics/bindings/generate_bindings.py`` and add your model to the ``main`` function in the generate_country or generate_canton function.

6. Edit ``CMakeLists.txt`` and add your model to the ``GENERATED_COUNTRY_BINDINGS`` variable.

7. Create a ``test/py/test_country_xyz.py`` analoguous to ``test_country_sir.h`` and test your code. You may skip testing the derivatives, since AD should already be tested.

In the case AD does not support some operation, add it in ``src/epidemics/utils/autodiff.h``.
Create a test in ``test/cpp/test_autodiff.cpp``!
