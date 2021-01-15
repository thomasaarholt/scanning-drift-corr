# scanning-drift-corr

Correction of nonlinear and linear scanning probe artifacts using orthogonal scan pairs.  

This method will also work on most time series, though it does not yet implement the fast scan direction correction algorithm.


# Running the code in Matlab

Running these command lines should reproduce the included example data in matlab.

First, load the data:
> load('data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat')

Next, verify the input data has horizontal fast scan directions (i.e. most drift artifacts occur in the vertical / row direction). This can also be easily done using windowed FFT images.
> figure('Name','Measured Data 0deg');imagesc(image00deg); axis equal off; colormap(gray(256));
> figure('Name','Measured Data 90deg');imagesc(image90deg); axis equal off; colormap(gray(256));
Zoom in on these images, note the row-by-row jumps and overall drift.

The first SPmerge step is to initialize the alignment structure, and perform a search over linear drift vectors (TODO: make this much faster via FFT shear).
> sMerge = SPmerge01linear([0 90],image00deg,image90deg);

Note that the alignment is overall quite poor and shows strong Moire artifacts. The yellow cross shows the reference position for the "wrinkle smoothing" initial alignment step. We should move it to a region of the image with good alignment before running the next step. For example:
> sMerge.ref = [560 450];

UPDATE - This new version of the code does not perform the "wrinkle smoothing" alignment steps by default!  In order to run some of these iterations, you must specify how many iterations you wish to perform using the third input argument. For example, try:
> sMerge = SPmerge02(sMerge,0,8);

Much better!  In the previous step, we ran 0 "final alignment" steps and 8 "wrinkle smoothing" steps.  We can now do the main alignment:
> sMerge = SPmerge02(sMerge);

The above command is equivalent to using sMerge = SPmerge02(sMerge,32); or sMerge = SPmerge02(sMerge,32,0);  Assuming the alignment has converged correctly, we can now generate a final output image:
> imageFinal = SPmerge03(sMerge);

Other notes - I've found that the new linear drift search finds a good initial alignment for most experimental datasets. This is the motivation behind defaulting to 0 iterations for the "wrinkle smoothing" step. However some long dwell time experiments will have slowly varying drift vectors, and so I've left this algorithm in this code.

The vast majority of bad alignments happen when the scan lines are not horizontal, or the images have a large amound of fast scan error (requires a different algorithm to correct). Most orthogonal image pairs can be well-aligned by playing around with this setting in SPmerge02.m:
originWindowAverage = 4;
Setting this value too large will render the algorithm unable to fit local origin shifts, and setting it too small could overfit to noise. The related value for the "wrinkle smoothing" algorithm is:
originInitialAverage = size(sMerge.scanLines,1) / 16;

The other important parameters to check are those that change the initial guess for the alignment. sMerge.ref for example, or the linear search steps in SPmerge01.m:
sMerge.linearSearch = linspace(-0.08,0.08,1+2*4);  

It's also worth pointing out that the algoithm requires enough padding around the images to fit all of the image translations - make sure this value in SPmerge01.m is not too small:
paddingScale = (1+1/2);

If you have image pairs that you are having difficulty with, please feel free to email them to me at clophus@lbl.gov and I would be happy to have a look!





## Authors

Colin Ophus

1. Original author of the Matlab code.
2. Wrote the drift correction paper.




## Publication

A publication describing this method can be found here:

https://doi.org/10.1016/j.ultramic.2015.12.002

