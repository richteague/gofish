# fishing.py

Fish for line detections by exploiting the known velocity structure of the disk.

## Background

[Yen et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...832..204Y/abstract) were the first to fully describe the method, although other comparable methods were also being used (e.g. [Teague et al., 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...592A..49T/abstract); [Matra et al., 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...842....9M/abstract)). The aim of this package is to allow users to readily replicate the approach in order to quickly and easily reproduce results using this method.

Much of the machinery for this is based on [`eddy`](https://github.com/richteague/eddy) ([Teague, 2019](https://ui.adsabs.harvard.edu/abs/2019JOSS....4.1220T/abstract)) which inverts this approach to use bright line emission to infer the velocity, rather then using the known velocity structure to extract weak line emission.

## Fishing in _uv_ Space

Working in the image-plane of data offers several advantages to working in the _uv_-plane, such as being able to mask specific spatial regions. However, working in the _uv_-plane can be much faster, better handle (or better yet, bypass entirely) uncertainties associated with spatial correlations, and does not need the inteferometric data to be imaged which can take both time and significant space.

We would strongly recommend the use of [`VISIBLE`](https://github.com/AstroChem/VISIBLE) ([Loomis et al., 2018](https://ui.adsabs.harvard.edu/abs/2018AJ....155..182L/abstract)) which uses match filtering to search for weak line emission _before_ trying any image-plane analysis.
