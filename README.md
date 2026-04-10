# ggplot2_py

AI-assisted Python port of the R **ggplot2** package — Create Elegant Data Visualisations Using the Grammar of Graphics.

## Overview

ggplot2_py implements the grammar of graphics in Python, faithfully porting R's ggplot2 using pandas DataFrames as the data container and a Cairo-based rendering backend. It supports 40+ geoms, 30+ stats, faceting, coordinate systems, themes, and scales.

## Dependencies

This package depends on three companion R-to-Python ports from the same project:

| Package | R source | Python import | Repository |
|---------|----------|---------------|------------|
| **grid_py** | `grid` | `grid_py` | [R2pyBioinformatics/grid_py](https://github.com/R2pyBioinformatics/grid_py) |
| **gtable_py** | `gtable` | `gtable_py` | [R2pyBioinformatics/gtable_py](https://github.com/R2pyBioinformatics/gtable_py) |
| **scales_py** | `scales` | `scales` | [R2pyBioinformatics/scales_py](https://github.com/R2pyBioinformatics/scales_py) |

Additional Python dependencies: numpy, pandas, matplotlib, scipy, pycairo.

## Installation

```bash
# Install companion packages first
pip install -e /path/to/grid_py
pip install -e /path/to/gtable_py
pip install -e /path/to/scales_py

# Install ggplot2_py
pip install -e ".[dev]"
```

## Quick Start

```python
from ggplot2_py import *
from ggplot2_py.datasets import mpg

(ggplot(mpg, aes(x="displ", y="hwy", colour="class"))
 + geom_point()
 + geom_smooth(method="lm")
 + facet_wrap("drv")
 + theme_minimal()
 + labs(title="Engine Displacement vs Highway MPG"))
```

## Tutorials

- [Getting Started](tutorials/ggplot2.ipynb) — core concepts: data, aes, geoms, stats, scales, facets, coords, themes
- [Geom Gallery](tutorials/geoms_gallery.ipynb) — boxplot, violin, density, tile, hex and combinations

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```
