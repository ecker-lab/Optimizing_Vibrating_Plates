import numpy as np
import torch
from scipy import ndimage
import cv2

DEFAULT_MIN_LENGTH_SCALE = 0.025

def get_x_angle(pattern, width):
    dx = float(width) / float(pattern.shape[1])
    dh_dx_pattern = np.diff(pattern, axis=1) / dx
    angle_x_pattern = np.arctan(dh_dx_pattern)
    angle_x_pattern = np.rad2deg(angle_x_pattern)
    return angle_x_pattern


def get_y_angle(pattern, height):
    dy = float(height) / float(pattern.shape[0])
    dh_dy_pattern = np.diff(pattern, axis=0) / dy
    angle_y_pattern = np.arctan(dh_dy_pattern)
    angle_y_pattern = np.rad2deg(angle_y_pattern)
    return angle_y_pattern


def check_derivative(pattern, max_angle=70, height=0.60, width=0.90):
    """
    pattern, height,width must have the same unit. By default the assumed unit is meter.
    """

    angle_y_pattern = get_y_angle(pattern, height)
    y_mask = angle_y_pattern < max_angle
    y_mask = np.pad(y_mask, [(0, 1), (0, 0)], mode="constant", constant_values=1)

    angle_x_pattern = get_x_angle(pattern, width)
    x_mask = angle_x_pattern < max_angle
    x_mask = np.pad(x_mask, [(0, 0), (0, 1)], mode="constant", constant_values=1)

    return x_mask & y_mask



def _calc_discretization(radius,bin_size):
    radius_rounded_up = np.ceil(radius/bin_size)*bin_size


    N = 2*np.ceil(radius_rounded_up/bin_size)
    if N%2==0:
        N+=1
    N = int(N)

    x = np.linspace(-radius_rounded_up,radius_rounded_up,N)
    return x

def get_structuring_element(radius,bin_width,bin_height):
    """
    Returns an structuring element, that is an circle.
        shape: (M,N)
        M,N are uneven
    """

    x = _calc_discretization(radius,bin_width)
    y = _calc_discretization(radius,bin_height)
    element = (y[:,None]**2+x[None,:]**2)<(radius+bin_width/2)**2
    return element.astype(np.uint8)



def check_beading_size(pattern,dimensions=(0.6,0.9),min_size=DEFAULT_MIN_LENGTH_SCALE, threshold=0.01):
    """
    pattern: array of shape (H,W)
    dimensions: (Height,Width), units in meter
    min_size: minimum size of beadings
    """
    if min_size<=0:
        return True
    
    bin_width = dimensions[1]/pattern.shape[1]
    bin_height = dimensions[0]/pattern.shape[0]

    element = get_structuring_element(min_size/2,bin_width,bin_height)


    mask = pattern > threshold
    mask = mask.astype(np.uint8)
    mask_open = cv2.morphologyEx(mask,cv2.MORPH_OPEN,element)

    result = mask_open == mask
    return result


def check_beading_space(pattern,dimensions=(0.6,0.9),min_space=DEFAULT_MIN_LENGTH_SCALE, threshold=0.01):
    """
    pattern: array of shape (H,W)
    dimensions: (Height,Width), units in meter
    min_space: minimum space between of beadings
    """
    if min_space<=0:
        return True
    
    bin_width = dimensions[1]/pattern.shape[1]
    bin_height = dimensions[0]/pattern.shape[0]

    element = get_structuring_element(min_space/2,bin_width,bin_height)

    mask = pattern > threshold
    mask = mask.astype(np.uint8)
    mask_close = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,element,borderValue=0)
    

    result = mask_close == mask
    return result


def check_boundary_condition(pattern):
    result = np.full(pattern.shape, fill_value=True)
    result[0, :] = np.isclose(pattern[0, :], 0)
    result[:, 0] = np.isclose(pattern[:, 0], 0)
    result[-1, :] = np.isclose(pattern[-1, :], 0)
    result[:, -1] = np.isclose(pattern[:, -1], 0)
    return result


def check_beading_height(pattern, atol=1e-6):
    return (pattern < 0.02 + atol) & (pattern > -atol)


def are_pixels_valid(pattern, dimensions=(0.6,0.9),min_beading_size=DEFAULT_MIN_LENGTH_SCALE,
                     min_beading_space=DEFAULT_MIN_LENGTH_SCALE, boundary_condition=True ):
    """
    patterns: is a numpy array of shape: B x H x W
        the units of the patterns are meters
    dimensions: (Height,Width)
    min_beading_size: the minimum beading size of a beadings
    min_beading_space: the minimum distance between beadings
    boundary_condition: wether to check if the pattern is zero at the boundary
    """

    mask = (
        check_derivative(pattern,height=dimensions[0],width=dimensions[1])
        & check_beading_height(pattern)
        & check_beading_space(pattern,dimensions,min_beading_space)
        & check_beading_size(pattern,dimensions,min_beading_size)
    )
    if boundary_condition:
        mask = mask & check_boundary_condition(pattern)

    return mask


def mean_valid_pixels( patterns, dimensions=(0.6,0.9),min_beading_size=DEFAULT_MIN_LENGTH_SCALE,
                      min_beading_space=DEFAULT_MIN_LENGTH_SCALE, boundary_condition=True):
    """
    patterns: is a numpy array of shape: B x H x W
        the units of the patterns are meters
    dimensions: (Height,Width)
    min_beading_size: the minimum beading size of a beadings
    min_beading_space: the minimum distance between beadings
    boundary_condition: weather the to check if the pattern is zero at the boundary
    """
    valid_pixels_means = []
    for pattern in patterns:
        valid_pixels_means.append(
            (are_pixels_valid(pattern, dimensions,min_beading_size,min_beading_space,boundary_condition)).mean()
        )
    return np.mean(valid_pixels_means)

def calc_beading_ratio(pattern):
    return pattern.sum()/(pattern.size*pattern.max())