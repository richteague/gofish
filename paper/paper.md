---
title: 'GoFish: Fishing for Line Observations in Protoplanetary Disks'
tags:
  - Python
  - astronomy
authors:
  - name: Richard Teague
    orcid: 0000-0003-1534-5186
    affiliation: 1
affiliations:
 - name: Department of Astronomy, University of Michigan, 1085 S. University Ave., Ann Arbor, MI 48109, USA
   index: 1
date: XX August 2019
bibliography: paper.bib
---

# Background

Molecular line observations are an essential complement to high resolution continuum data when studying the planet formation environment, the protoplanetary disk. However, unlike continuum observations, which can exploit the large bandwidth of current telescopes, observations of line emission requirer a much higher sensitivity for them to be robustly detected. A common approach in astronomy to deal with this issue is to stack (average) multiple observations such that the noise, which is assumed to random, will cancel out, leaving a stronger detection of the line.

For the case of a protoplanetary disk (or any source which has a significant level of rotation), the rotation will Doppler shift the lines at a given location to a slightly offset frequency. For example, for a disk rotating with a rotation profile $v_{\rm rot}(r)$, the projected line of sight velocity (and thus offset in the Doppler shifted line center), at a given disk-centric coordinate $(r,\, \theta)$ where $\theta$ is the polar angle in the disk, is given by,

$$\delta v(r,\, \theta) = v_{\rm rot}(r) \cos (\theta) \sin (i)$$

where $i$ is the inclination of the disk. Thus, if $v_{\rm rot}(r)$ is known, then this shift can be accounted for by 'correcting' each spectrum before stacking.

This method was first described in `@Yen:2016`, however other groups were using similar techniques, such as `@Teague:2016` and `@Matra:2017`. A recently published code, `eddy` `[@Teague:2019]`, inverts this method to use strongly detected line emission in infer the rotation profile.

# Code Summary

The aim of `GoFish` is provide the functionality to perform such analyses and make them easily reproducible. The user only needs an image cube in the common FITS format, where two axes are spatial dimensions and the third is spectral.

Using known properties of the disk, namely the inclination and position angle of the disk and the distance and mass of the mass of the central star, averaged or integrated spectra over a specified radial range can be extracted. In addition, a convenience function `find_center` is provided which searches for the disk center based on maximising the signal-to-noise ratio of the extracted spectrum. This is particularly helpful if there is no clear detection of the line in the raw images.

The code is fully documented, including Juypter Notebook tutorials to demonstrate the functions and their syntax. These are run using publicly available data cubes and are thus easy to replicate.

# Acknowledgements

I would like to thank Ryan Loomis for helpful discussions on the implementation of this method. R.T. was supported by NSF grant AST-1514670 and NASA NNX16AB48G.
