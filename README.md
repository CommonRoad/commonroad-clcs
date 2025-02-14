# CommonRoad Curvilinear Coordinate System (CLCS)

Curvilinear coordinate frames (also Frenet frames) are a widely used representation in motion planning for automated
vehicles. Curvilinear frames are aligned with a reference path and Cartesian points (x, y) are described by 
coordinates (s, d), where s is the arc length and d the lateral deviation to the reference path.
The coordinate transformation is performed via a projection on the reference path for which different methods can be 
used [1] [2].

<img src="docs/assets/animation.gif" alt="clcs" width="500"/>

This project software for constructing and using curvilinear coordinate frames for reference paths given as polylines.
We offer:
* C++ backend for efficient coordinate transformations
* Python frontend for usage with CommonRoad scenarios
* various utility methods for processing reference path (e.g., smoothing, curvature reduction, resampling etc ...)

## :wrench: Installation
We provide two installation options for CommonRoad-CLCS: 
Installation as a Python package (recommended) or building from source.

1. **Python Package**: Install the python package via `pip` in your Conda environment:
    ```bash
    pip install commonroad-clcs
    ```
2. **Build from source**: To build the project from source and install it in your Conda environment, 
   please refer to the [README_FOR_DEVS](./README_FOR_DEVS.md).
   This option is only recommended for advanced users who would like to use or modify the C++ backend directly.



## :link: System Requirements
The software is written in Python 3.10 and C++17, and was tested on Ubuntu 20.04 and 22.04.
It should be compatible with later versions.
For building the code, the following minimum versions are required:
* **GCC and G++**: version 10 or above
* **CMake**: version 3.28 or above.
* **Pip**: version 24.2 or above

We recommend using [Anaconda](https://www.anaconda.com/) as a package manager for Python.


## :rocket: Getting Started
To get started, please see the minimal example in `./tutorials/clcs_minimal_example.py`.

For more details, please refer to the Jupyter notebook tutorials provided in `./tutorials/`. 


## :books: Documentation
A documentation including APIs is available on our website at:
https://cps.pages.gitlab.lrz.de/commonroad/commonroad-clcs/.

To build the documentation locally, please install the optional `docs` listed in [pyproject.toml](pyproject.toml) first.
The documentation can then be built using mkdocs via:
```bash
mkdocs build
```
You can browse the doc by launching `./docs/site/index.html`

## :busts_in_silhouette: Authors
**Contributors** (alphabetic order by last name): Peter Kocsis, Edmond Irani Liu,
Stefanie Manzinger, Tobias Markus, Evald Nexhipi, Vitaliy Rusinov, Daniel Tar, Gerald Würsching

## :speech_balloon: References
If you use our software for research, please cite:
```
@inproceedings{wursching2024robust,
    author={W{\"u}rsching, Gerald and Althoff, Matthias},
    title={Robust and Efficient Curvilinear Coordinate Transformation with Guaranteed Map Coverage for Motion Planning},
    booktitle={Proc. of the  IEEE Intelligent Vehicles Symposium},
    year={2024}
}
```

#### Additional references:
[1] [Héry, Elwan, Stefano Masi, Philippe Xu, and Philippe Bonnifait. "Map-based curvilinear coordinates for autonomous vehicles." ITSC, 2017](https://ieeexplore.ieee.org/document/8317775)

[2] [Bender, Philipp, Julius Ziegler, and Christoph Stiller. "Lanelets: Efficient map representation for autonomous driving." IV, 2014](https://ieeexplore.ieee.org/document/6856487)
