# Mosaiq for Python

This is a simplified [mosaic plot](https://en.wikipedia.org/wiki/Mosaic_plot)
technique that works for numeric/categorical data.

![Imgur](https://i.imgur.com/atssMvU.png)

For categorical data, a frequency table of values is calculated.  Only the top 7
most common categories are preserved.  The rest are replaced by "TOP_NA".

For numeric data, a histogram is calculated over the distribution. The precise
numeric values are replaced by its respective bin.



