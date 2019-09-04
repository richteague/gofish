# GoFish
<bR>
<p align='center'>
  <br/>
  <img src="https://github.com/richteague/gofish/blob/master/docs/_static/logo.png" width="596" height="527"><br/>
  <br>
  <b>Fishing for molecular line detections in protoplanetary disks.</b>
  <br>
  For more information, read the documentation.
  <br>
  <br>
  <a href="http://joss.theoj.org/papers/f2808d0c1cc0ffb51aa60466c896ed06">
      <img src="http://joss.theoj.org/papers/f2808d0c1cc0ffb51aa60466c896ed06/status.svg"></a>
  <a href='https://fishing.readthedocs.io/en/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/fishing/badge/?version=latest' alt='Documentation Status' />
  </a>
</p>

<br>

________________


### Installation

The quickest way to install is using PyPI:

`> pip install gofish`

which will install all the necessary dependancies.

### Example Usage

For a thorough introduction on how to use `GoFish`, see the extensive [documentation](https://fishing.readthedocs.io/en/latest/).

In brief, the user will attach an image cube to `GoFish` which will read the necessary header information:

```python
# Attach an image cube.
from gofish import imagecube
cube = imagecube('path/to/cube.fits')
```

Once attached averaged spectrum over a user-specified region can be extracted using the known geometrical properties of the disk:

```python
# Return the averaged spectrum between 0.0" and 1.0".
x, y, dy = cube.average_spectrum(r_min=0.0, r_max=1.0, inc=5.0,
                                 PA=152., mstar=0.88, dist=59.5)
```

where `x` is the velocity axis, `y` is the spectrum and `dy` is the uncertainty. Alternatively the integrated spectrum can be extracted in a similar manner,

```python
# Return the integrated spectrum between 0.0" and 1.0".
x, y, dy = cube.integrated_spectrum(r_min=0.0, r_max=1.0, inc=5.0,
                                    PA=152., mstar=0.88, dist=59.5)
```

where `y` is now the integrated flux in units of Jy.

### Citation

Coming soon.
