# Notes for Developers

Here we provide more detailed information for using our software for development purposes.


## Table of Contents
- [Building from Source](#building-from-source)
- [Editable Installation](#editable-installation)
- [Debugging with an IDE](#debugging-with-an-ide)
- [Running Unit Tests](#running-unit-tests)


## Building from Source

### Build dependencies
**Manual installation:**
- [OpenMP](https://www.openmp.org/)

**Optional installation:** The following dependencies are automatically installed via CMake FetchContent.
However, to speed up the build we recommend to optionally install them manually.
- [Boost.Geometry](https://www.boost.org/doc/libs/1_79_0/libs/geometry/doc/html/index.html)
- [Eigen3](https://eigen.tuxfamily.org/)
- [pybind11](https://github.com/pybind/pybind11)
- [spdlog](https://github.com/gabime/spdlog)

The additional Python build dependencies are listed in [pyproject.toml](pyproject.toml) under `build`.


### Building the project
We use scikit-build-core to build the C++ code via pip

1. Install the aforementioned build dependencies.
2. Build the package via
   ```bash
   pip install -v .
   ```
   This will be build the C++ code, the python binding shared library (pycrccosy*.so) and install the
   package in you conda environment.


## Editable Installation
1. Install the aformenetioned C++ dependencies. 

2. Install the Python build dependencies (required to make `--no-build-isolation` work in the next step):
```bash
pip install -r requirements_build.txt
```

3. Build the package and install it in editable mode with automatic rebuilds.
```bash
pip install -v --no-build-isolation --config-settings=editable.rebuild=true -e .
```

Please check the [scikit-build-core documentation](https://scikit-build-core.readthedocs.io/en/latest/configuration.html#editable-installs) for more details.

Flags:
- `-v` (verbose) output about the build progress
- `--no-build-isolation` disables build isolation, build runs in your local environment
- `--config-settings=editable.rebuild=true` enables automatic rebuilds when the source code changes
- `-e` editable install 


## Debugging the code
1. Install in editable mode using the CMake Debug build flag:
```bash
pip install -v --no-build-isolation --config-settings=editable.rebuild=true --config-settings=cmake.build-type="Debug" -e .
```

2. Launch the Python interpreter together with a C++ debugger (e.g., GDB):
```bash
gdb -ex r --args python compute_reachable_set.py
```

## Building Python Bindings Directly

Building the Python bindings directly, i.e., without using scikit-build-core, can be helpful e.g. to set up your IDE.
To do so, you need to add the following parameters to your CMake invocation.
```
-DCR_CLCS_BUILD_PYTHON_BINDINGS=ON
-DCMAKE_PREFIX_PATH=/path/to/site-packages
```
The first parameter enables the Python bindings for the CLCS.
The second parameters adds the path to your Python installation's `site-packages` directory to the CMake search path like scikit-build-core does.
If you are using an Anaconda/Miniconda environment, make sure to point this to the `site-packages` directory of the correct environment.
Please make sure that `pybind11` with the version specified in the `build-system.requires` is installed in this environment.


## Running Unit Tests

Python unit tests are located in `./tests/` and can be run via pytest.
