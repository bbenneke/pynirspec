# -*- coding: utf-8 -*-

"""A module for various utilities and helper functions"""

import numpy as np

DTYPEf = np.float64

DTYPEi = np.int32

def replace_nans(array,method='localmean'):
        
    # indices where array is NaN
    inans, jnans = np.nonzero( np.isnan(array) )
    
    # number of NaN elements
    n_nans = len(inans)
    
    # arrays which contain replaced values to check for convergence
    replaced_new = np.zeros( n_nans, dtype=DTYPEf)
    replaced_old = np.zeros( n_nans, dtype=DTYPEf)
    
    # depending on kernel type, fill kernel array
    if method == 'localmean':
        kernel_size = 1
        shape = (2*kernel_size+1,2*kernel_size+1)
        kernel = np.ones(shape)

    elif method == 'idw':
        kernel_size = 2
        kernel = np.array([[0, 0.5, 0.5, 0.5,0],
                  [0.5,0.75,0.75,0.75,0.5], 
                  [0.5,0.75,1,0.75,0.5],
                  [0.5,0.75,0.75,0.5,1],
                  [0, 0.5, 0.5 ,0.5 ,0]])
    else:
        raise ValueError( 'method not valid. Should be one of `localmean`.')
    
    # fill new array with input elements
    filled = array.copy()

    for k in range(n_nans):
        i = inans[k]
        j = jnans[k]
            
        # initialize to zero
        filled[i,j] = 0.0
        n = 0

        # if we are not out of the boundaries
        if i+kernel_size < array.shape[0] and i-kernel_size >= 0:
            if j+kernel_size < array.shape[1] and j-kernel_size >= 0:
                subarray = filled[i-kernel_size:i+kernel_size+1,j-kernel_size:j+kernel_size+1]
                gsubs = np.where(np.isfinite(subarray))
                if len(gsubs) > 0:
                    filled[i,j] = subarray[gsubs].sum()
                    neff = kernel[gsubs].sum()-1.
                    filled[i,j] /= neff
        
    
    return filled


def sincinterp(image, x,  y, kernel_size=3 ):
    """
    Re-sample an image at intermediate positions between pixels.
    This function uses a cardinal interpolation formula which limits
    the loss of information in the resampling process. It uses a limited
    number of neighbouring pixels.
    The new image :math:`im^+` at fractional locations :math:`x` and :math:`y` is computed as:
    .. math::
    im^+(x,y) = \sum_{i=-\mathtt{kernel\_size}}^{i=\mathtt{kernel\_size}} \sum_{j=-\mathtt{kernel\_size}}^{j=\mathtt{kernel\_size}} \mathtt{image}(i,j) sin[\pi(i-\mathtt{x})] sin[\pi(j-\mathtt{y})] / \pi(i-\mathtt{x}) / \pi(j-\mathtt{y})
    Parameters
    ----------
    image : np.ndarray, dtype np.int32
    the image array.
    x : two dimensions np.ndarray of floats
    an array containing fractional pixel row
    positions at which to interpolate the image
    y : two dimensions np.ndarray of floats
    an array containing fractional pixel column
    positions at which to interpolate the image
    kernel_size : int
    interpolation is performed over a ``(2*kernel_size+1)*(2*kernel_size+1)``
    submatrix in the neighbourhood of each interpolation point.
    Returns
    -------
    im : np.ndarray, dtype np.float64
    the interpolated value of ``image`` at the points specified
    by ``x`` and ``y``
    """
       
    # the output array
    r = np.zeros( [x.shape[0], x.shape[1]], dtype=DTYPEf)
          
    # fast pi
    pi = 3.1419
        
    # for each point of the output array
    for I in range(x.shape[0]):
        for J in range(x.shape[1]):
            
            #loop over all neighbouring grid points
            for i in range( int(x[I,J])-kernel_size, int(x[I,J])+kernel_size+1 ):
                for j in range( int(y[I,J])-kernel_size, int(y[I,J])+kernel_size+1 ):
                    # check that we are in the boundaries
                    if i >= 0 and i <= image.shape[0] and j >= 0 and j <= image.shape[1]:
                        if (i-x[I,J]) == 0.0 and (j-y[I,J]) == 0.0:
                            r[I,J] = r[I,J] + image[i,j]
                        elif (i-x[I,J]) == 0.0:
                            r[I,J] = r[I,J] + image[i,j] * np.sin( pi*(j-y[I,J]) )/( pi*(j-y[I,J]) )
                        elif (j-y[I,J]) == 0.0:
                            r[I,J] = r[I,J] + image[i,j] * np.sin( pi*(i-x[I,J]) )/( pi*(i-x[I,J]) )
                        else:
                            r[I,J] = r[I,J] + image[i,j] * np.sin( pi*(i-x[I,J]) )*np.sin( pi*(j-y[I,J]) )/( pi*pi*(i-x[I,J])*(j-y[I,J]))
    return r



