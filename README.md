# mdwtools
201-10-30

Base set of functions for reading and plotting GCM output, tailored to CESM, along with some comparison datasets. The main files are mdwfunctions, which contains a bunch of loading and analysis functions, and mdwtools, which contains a bunch of plotting functions. I have attempted to make this into a package and have a nice setup file, but you can get around needing that with softlinking as this is still very much under active development. Much of the code was written to work with my personal netCDF class (as defined in mdwfunctions), but I am transitioning the code in this repository to work with xarray instead. Some things work with xarray, and some don't at this point.
