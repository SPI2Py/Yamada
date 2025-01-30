# Yamada: The Python Library for Calculating the Yamada Polynomial of Spatial Graphs

[![ASME IDETC Paper](https://img.shields.io/badge/DOI-10.1038%2Fs41592--019--0686--2-blue)](
https://doi.org/10.1115/DETC2021-66900)

[![Python package](https://github.com/Chad-Peterson/Yamada/actions/workflows/tests.yml/badge.svg)](https://github.com/Chad-Peterson/Yamada/actions/workflows/tests.yml)

[![Windows](https://svgshare.com/i/ZhY.svg)](https://svgshare.com/i/ZhY.svg)
[![macOS](https://svgshare.com/i/ZjP.svg)](https://svgshare.com/i/ZjP.svg)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)


[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)


## Spatial Topologies and Yamada Polynomials

Systems such as automotive cooling layouts, hybrid-electric power trains, and aero-engines are made up of interconnected
components that are spatially arranged to meet system requirements. Holistically optimizing these types of systems is an
extremely challenging problem due to the combinatorial nature of the design space. The research community is exploring
different design representations and algorithms to address this problem.

This library provides a Python implementation of the spatial graphs, spatial graph diagrams, and Yamada polynomial.
These spatial-topological constructs are powerful tools for representing and analyzing complex engineering systems.
By representing engineering systems as spatial topologies we abstract away complex geometry while
retaining some low-fidelity, directionally correct information. Yamada polynomials are a calculated quantity that
is essentially a fingerprint of a spatial topology. This fingerprint can be used to identify unique spatial topologies.

We are currently collecting and analyzing empirical data to determine the effectiveness of spatial topologies and
Yamada polynomials as a design representation for different classes of problems.

## Important Notice

Since this library is still early in development features are often added and removed.
Please feel free to reach out to Chad <cp44@illinois.edu> with any questions or concerns.

## Installation
Yamada requires Python 3.9+ and is supported on Windows, Mac, and Linux. 
It can be installed from PyPI with the following command in your terminal:

>pip install yamada

## Cite Us

@software{Yamada2023github,
  author = {Chad Peterson, Nathan Dunfield},
  title = {Yamada: The Python Library for Calculating the Yamada Polynomial of Spatial Graphs},
  url = {https://github.com/Chad-Peterson/Yamada},
  version = {0.2.1},
  year = {2023},
}



