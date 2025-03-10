# Mosaiq

The Mosaiq library introduces `mosaiq` as a Python function that generates a [mosaic](https://en.wikipedia.org/wiki/Mosaic_plot) plot using [Altair](https://altair-viz.github.io/), supporting both categorical and numeric fields. This versatile visualization tool automatically bins numeric data and consolidates low-frequency categories into a single "TOP_NA" group. The plot is designed to provide a clear overview of the distribution and relationship between two fields, with customizable color schemes.

## Features

- **Automatic Binning for Numeric Data**: Uses histogram binning for numeric fields based on a specified maximum number of bins.
- **Top-N Categories for Categorical Data**: Keeps only the most frequent categories (up to `max_bins`), combining all others into a "TOP_NA" bin.
- **Customizable Color Scheme**: A single `color` parameter controls the color scheme for both fields, allowing a unified look across all bins.
- **Tooltip Support**: Hovering over bins displays detailed information, including category labels and counts.

## Dependencies

- [Altair](https://altair-viz.github.io/) for visualization.
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation.
- [Narwhals](https://github.com/narwhals/narwhals) for handling specific typing.

## Installation

Install the required Python packages with:

```bash
pip install mosaiq 
```

## Usage

### Function Signature

```python
mosaiq(dataframe: FrameT, field1: str, field2: str, max_bins=6, color="category20")
```

### Parameters

- **dataframe** (`FrameT`): A pandas DataFrame containing the data to be visualized.
- **field1** (`str`): Name of the first field (categorical or numeric) to display on the x-axis.
- **field2** (`str`): Name of the second field (categorical or numeric) to display as blocks within the mosaic.
- **max_bins** (`int`, optional): Maximum number of bins or categories to display. Defaults to `6`.
- **color** (`str`, optional): Color scheme for all bins. Defaults to `"category20"`.
- **top_na_label** (`str`, optional): Provide a custom label for category bins that do not pass max_bins threshold. 

### Returns

- **altair.Chart**: A compound Altair chart representing the mosaic plot.

### Example Usage


```python
from vega_datasets import data
mosaiq(data.seattle_weather(), "weather", "wind").configure_view(continuousWidth=900)
```

![Seattle Weather](https://github.com/jdonaldson/mosaiq-python/blob/master/img/seattle_example.png?raw=true)

```python
import pandas as pd
import narwhals as nw
from mosaiq import mosaiq

# Create a sample DataFrame
data = {
    "Category": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "Value": [10, 15, 7, 30, 45, 10, 22, 5]
}
df = pd.DataFrame(data)

# Generate a mosaic plot
chart = mosaiq(df, "Category", "Value", max_bins=5, color="blueorange")
chart.display()
```

![Color Argument Example](https://github.com/jdonaldson/mosaiq-python/blob/master/img/color_example.png?raw=true)

## Customization

- **Adjust Binning**: Control the number of bins for numeric fields with `max_bins`. If more categories than `max_bins` are present, the function groups the least frequent categories into a new "TOP_NA" category.
- **Color Scheme**: Set a color scheme using any valid Altair color scheme name (e.g., `"blues"`, `"viridis"`, `"category10"`). This single color parameter unifies the plot’s appearance.

## Additional Notes

This function is decorated with `@nw.narwhalify` to handle non-pandas DataFrame input using Narwhals typing. If you’re unfamiliar with Narwhals, check out the [Narwhals GitHub repo](https://github.com/narwhals/narwhals) for further information.

## License

MIT License. See `LICENSE` for more information.

---

Enjoy exploring your data with `Mosaiq`!
