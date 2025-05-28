import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from torch.distributions import Uniform
from torch.utils.data import Dataset
from scipy import ndimage
from plate_optim.metrics.manufacturing import (
    get_structuring_element,
    DEFAULT_MIN_LENGTH_SCALE,
)
import cv2

def rot_mat(theta):
    theta_rad = theta * math.pi / 180
    matrix = [
        [math.cos(theta_rad), -math.sin(theta_rad)],
        [math.sin(theta_rad), math.cos(theta_rad)],
    ]
    return matrix


def draw_line(draw, max_length, resolution, w_lines, length="sampled"):
    angle = Uniform(0, 180).sample()
    if length == "sampled":
        length = Uniform(0, max_length).sample()
    else:
        length = max_length
    width = int(np.random.uniform(w_lines[0], w_lines[1]) * resolution[0] / 100)
    line = np.array([((-length / 2, 0.0), (length / 2, 0.0))]) @ rot_mat(angle)

    delta_x = np.random.uniform(resolution[0])
    line[:, :, 0] = line[:, :, 0] + delta_x
    delta_y = np.random.uniform(0, resolution[1])
    line[:, :, 1] = line[:, :, 1] + delta_y

    draw.line(
        (line[0, 0, 0], line[0, 0, 1], line[0, 1, 0], line[0, 1, 1]),
        fill="white",
        width=width,
    )
    return draw


def draw_ellipse(draw, img, w_ellipses, resolution, length_ellipse):
    width1, width2, width3 = (np.random.uniform(w_ellipses[0], w_ellipses[1], 3) * resolution[0] / 100)
    x_mid = resolution[0] + np.random.uniform(0, resolution[0])
    y_mid = resolution[1] + np.random.uniform(0, resolution[1])
    length_x, length_y = np.random.uniform(
        length_ellipse[1] * resolution[0] / 100,
        resolution[0] - length_ellipse[0] * resolution[0] / 100,
        2,
    )

    img_ellipse = Image.new("L", (3 * resolution[0], 3 * resolution[1]), color=(0))
    draw_ellipse_element = ImageDraw.Draw(img_ellipse)

    x = np.array([x_mid - length_x / 2, y_mid - length_y / 2])
    y = np.array([x_mid + length_x / 2, y_mid + length_y / 2])
    xy = [x[0], x[1], y[0], y[1]]

    draw_ellipse_element.ellipse(xy, fill="white", outline="white", width=width3)

    x2 = x + width1
    y2 = y - width2
    draw_ellipse_element.ellipse([x2[0], x2[1], y2[0], y2[1]], fill="black", outline="black")

    angle = np.random.uniform(0, 90, 1)
    img_ellipse = img_ellipse.rotate(angle)
    img_ellipse = img_ellipse.crop((resolution[0], resolution[1], 2 * resolution[0], 2 * resolution[1]))
    img = ImageChops.lighter(img, img_ellipse)
    return draw, img


def draw_bounding_box(img, resolution, width):
    fill_color = "black"
    p1 = (0, 0)
    p2 = (0, resolution[1])
    p3 = (resolution[0], resolution[1])
    p4 = (resolution[0], 0)
    p5 = p1 + width
    p6 = (0 + width, resolution[1] - width)
    p7 = p3 - width
    p8 = (resolution[0] - width, 0 + width)
    left_box = np.array([p1, p2, p6, p5])
    upper_box = np.array([p2, p3, p7, p6])
    right_box = np.array([p3, p4, p8, p7])
    lower_box = np.array([p1, p5, p8, p4])
    boxes = [left_box, upper_box, right_box, lower_box]
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([tuple(p) for p in box], fill=fill_color, width=0, outline=fill_color)
    return img


def draw_simple_img(
    resolution=[121, 81],
    n_lines=[1, 2],
    w_lines=[4, 7],
    n_ellipses=[0, 2],
    w_ellipses=[3, 5],
    gauss_blur=[1.05, 1.45],
    length_ellipse=[10, 15],
):
    img = Image.new("L", (resolution[0], resolution[1]), color=(0))
    draw = ImageDraw.Draw(img)
    max_length = np.sqrt(np.square(resolution[0]) + np.square(resolution[1]))
    n_lines_plot = np.random.randint(n_lines[0], n_lines[1] + 1)
    for i in range(n_lines_plot):
        draw = draw_line(draw, max_length, resolution, w_lines, length="fixed")

    n_ellipses_plot = np.random.randint(n_ellipses[0], n_ellipses[1] + 1)
    for i in range(n_ellipses_plot):
        draw, img = draw_ellipse(draw, img, w_ellipses, resolution, length_ellipse)
    img = draw_bounding_box(img, resolution, width=np.array(5 * resolution[0] / 100))
    img = img.filter(ImageFilter.GaussianBlur(np.random.uniform(*gauss_blur)))
    return np.array(img) / 255 * 0.02


def draw_line_P1P2(img, P1, P2, width=4, resolution_factor=1):

    resolution = np.array([img.width, img.height])
    width = width * resolution_factor
    fill_color = 'white'

    # Shift lines horizontal and vertical
    img_line = Image.new('L', (resolution[0], resolution[1]), color=(0))
    draw = ImageDraw.Draw(img_line)
    draw.line([(P1[0], P1[1]), (P2[0], P2[1])], fill=fill_color, width=width)

    img = ImageChops.lighter(img, img_line)
    return img


def draw_rectangle_hole(img, length_x=15, length_y=15, x_mid=None, y_mid=None, margin=None, angle=0.0, resolution_factor=1.0):

    resolution = np.array([img.width, img.height])

    if x_mid is None: x_mid = resolution[0] / 2
    if y_mid is None: y_mid = resolution[1] / 2

    length_x = length_x * resolution_factor
    length_y = length_y * resolution_factor
    x_mid = resolution[0] + x_mid * resolution_factor  # Offsest resoltion[0] since the Image is drawn on 3*resolution[0], 3*resolution[1]
    y_mid = resolution[1] + y_mid * resolution_factor  # Offsest resoltion[1] since the Image is drawn on 3*resolution[0], 3*resolution[1]

    img_rect = Image.new('L', (3 * resolution[0], 3 * resolution[1]), color=(0))
    draw_rect = ImageDraw.Draw(img_rect)

    x = np.array([x_mid - length_x / 2, y_mid - length_y / 2])
    y = np.array([x_mid + length_x / 2, y_mid + length_y / 2])
    top_left = (x[0], x[1])
    bottom_right = (y[0], y[1])

    width = y[0] - x[0]
    height = y[1] - x[1]

    draw_rect.rectangle([top_left, bottom_right], outline="white", fill="white")

    if margin is not None:
        border = np.min(np.array([width, height])) * margin
        img_rect_neg = Image.new('L', (3 * resolution[0], 3 * resolution[1]), color=(0))
        draw_rect_neg = ImageDraw.Draw(img_rect_neg)
        top_left_update = top_left + border
        bottom_right_update = bottom_right - border
        draw_rect_neg.rectangle([(top_left_update[0], top_left_update[1]), (bottom_right_update[0], bottom_right_update[1])],
                                outline="white",
                                fill="white")
        img_rect = ImageChops.subtract(img_rect, img_rect_neg)

    img_rect = img_rect.rotate(angle)
    img_rect = img_rect.crop((resolution[0], resolution[1], 2 * resolution[0], 2 * resolution[1]))
    img = ImageChops.lighter(img, img_rect)
    return img


def draw_arc(img, length_x=30, length_y=30, x_mid=None, end_angle=360, y_mid=None, angle=0.0, width=5, resolution_factor=1):

    resolution = np.array([img.width, img.height])
    if x_mid is None: x_mid = resolution[0] / 2
    if y_mid is None: y_mid = resolution[1] / 2

    width = width * resolution_factor
    length_x = length_x * resolution_factor
    length_y = length_y * resolution_factor
    x_mid_ellipse = resolution[
        0] + x_mid * resolution_factor  # Offsest resoltion[0] since the Image is drawn on 3*resolution[0], 3*resolution[1]
    y_mid_ellipse = resolution[
        1] + y_mid * resolution_factor  # Offsest resoltion[1] since the Image is drawn on 3*resolution[0], 3*resolution[1]

    img_arc = Image.new('L', (3 * resolution[0], 3 * resolution[1]), color=(0))
    draw_arc = ImageDraw.Draw(img_arc)

    x_mid = x_mid_ellipse
    y_mid = y_mid_ellipse

    x = np.array([x_mid - length_x / 2, y_mid - length_y / 2])
    y = np.array([x_mid + length_x / 2, y_mid + length_y / 2])
    xy = [x[0], x[1], y[0], y[1]]

    draw_arc.arc(xy, start=0.0, end=end_angle, fill="white", width=width)

    img_arc = img_arc.rotate(angle)
    img_arc = img_arc.crop((resolution[0], resolution[1], 2 * resolution[0], 2 * resolution[1]))

    img = ImageChops.lighter(img, img_arc)
    return img


def mirror_lr(img):

    # Get the width and height of the image
    width, height = img.size

    # Calculate the midpoint of the image
    midpoint = width // 2

    # Crop the left half of the image
    left_half = img.crop((0, 0, midpoint, height))

    # Mirror the left half
    mirrored_left_half = left_half.transpose(Image.FLIP_LEFT_RIGHT)

    # Create a new image with the mirrored left half and the original right half
    mirrored_image = Image.new('L', (width, height), color=(0))
    mirrored_image.paste(left_half, (0, 0))
    mirrored_image.paste(mirrored_left_half, (midpoint, 0))
    return mirrored_image


def mirror_ud(img):

    # Get the width and height of the image
    width, height = img.size

    # Calculate the midpoint of the image
    midpoint = height // 2

    # Crop the left half of the image
    up_half = img.crop((0, 0, width, midpoint))

    # Mirror the left half
    mirrored_up_half = up_half.transpose(Image.FLIP_TOP_BOTTOM)

    # Create a new image with the mirrored left half and the original right half
    mirrored_image = Image.new('L', (width, height), color=(0))
    mirrored_image.paste(up_half, (0, 0))
    mirrored_image.paste(mirrored_up_half, (0, midpoint))
    return mirrored_image


def mirror_quater(img):

    # Get the width and height of the image
    width, height = img.size

    # Calculate the midpoint of the image
    midpoint_width = width // 2
    midpoint_height = height // 2

    # Crop the left half of the image
    top_left = img.crop((0, 0, midpoint_width, midpoint_height))

    # Mirror the left half
    mirrored_top_right = top_left.transpose(Image.FLIP_LEFT_RIGHT)
    mirrowed_bottom_left = top_left.transpose(Image.FLIP_TOP_BOTTOM)
    mirrowed_bottom_right = mirrored_top_right.transpose(Image.FLIP_TOP_BOTTOM)

    # Create a new image with the mirrored left half and the original right half
    mirrored_image = Image.new('L', (width, height), color=(0))
    mirrored_image.paste(top_left, (0, 0))
    mirrored_image.paste(mirrored_top_right, (midpoint_width, 0))
    mirrored_image.paste(mirrowed_bottom_left, (0, midpoint_height))
    mirrored_image.paste(mirrowed_bottom_right, (midpoint_width, midpoint_height))
    return mirrored_image


def apply_engineering_blur(image, height_mat):

    #erode first the image, so the beading do not get wider,
    #because of the engineering kernel
    beading_height = height_mat.max()
    threshold = beading_height/2
    kernel = height_mat>threshold


    image_mat = ndimage.binary_erosion(image>threshold,kernel)
    image_mat = np.array(image_mat, dtype=float)

    height_mat = height_mat / np.max(height_mat.flatten())  
    max_val = height_mat.max()
    filter_img = ndimage.grey_dilation(image_mat, structure=height_mat)
    filter_img -= max_val

    filter_img*=beading_height
    return filter_img

def _draw_circle_kernel(radius,pixel_width,pixel_height):
    x = np.arange(-radius,radius,step=pixel_width)
    y = np.arange(-radius,radius,step=pixel_height)

    xx,yy = np.meshgrid(x,y)

    kernel = (xx**2+yy**2)<(radius**2)

    return kernel.astype(int)    


def apply_length_scale_constraint(image,min_length_scale,img_dimensions,max_height=0.02):
    if min_length_scale is None:
        return image
    if min_length_scale<1e-6:
        return image

    #adding one 1cm to the kernel to counteract the erosion of the engineering kernel
    min_length_scale+=0.01

    threshold = max_height/2 

    image = np.array(image)

    pixel_height = img_dimensions[0]/image.shape[0]
    pixel_width = img_dimensions[1]/image.shape[1]



    kernel = get_structuring_element(min_length_scale/2,pixel_width,pixel_height)
    #kernel[:,:]=1
    mask = (image>threshold).astype(np.uint8)

    pad_h,pad_w =kernel.shape
    mask[:pad_h,:]=0
    mask[-pad_h:,:]=0
    mask[:,:pad_w]=0
    mask[:,-pad_w:]=0

    #applying a erosion and a dilatation of same size
    #this removes unwanted spots in the background
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

    #applies a dilatation and then a erosion
    #this remove the unwanted spots in the forground
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,borderValue=0)

    #Alternative approach:
    #This removes a lot of beadings 
    #so I don't use it

    #mask = cv2.erode(mask,kernel=kernel_close,borderType=1)
    #mask = cv2.erode(mask,kernel=kernel_close)
    #mask = cv2.dilate(mask,kernel=kernel_open)

    return mask.astype(float)*max_height



def get_height_mat(img, height=0.02):
    # Convert to array
    img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
    beading_mat = np.asarray(img_flipped)

    # Assign height
    # print(np.max(beading_mat.flatten()))
    if (np.max(beading_mat.flatten()) != 0.0):
        beading_mat = beading_mat / np.max(beading_mat.flatten()) * height
    return beading_mat


def apply_gaussian_blur(img, gauss_blur=1.0, resolution_factor=1):
    img = img.filter(ImageFilter.GaussianBlur(gauss_blur * resolution_factor))
    return img


def draw_bounding_box_2(img, width=10, resolution_factor=1):

    width = np.float64(width * resolution_factor)
    resolution = np.array([img.width, img.height])

    fill_color = 'black'

    P1 = (0, 0)
    P2 = (0, resolution[1])
    P3 = (resolution[0], resolution[1])
    P4 = (resolution[0], 0)
    P5 = P1 + (width - 1)
    P6 = (0 + (width - 1), resolution[1] - width)
    P7 = P3 - width
    P8 = (resolution[0] - width, 0 + width)

    left_box = np.array([P1, P2, P6, P5])
    upper_box = np.array([P2, P3, P7, P6])
    right_box = np.array([P3, P4, P8, P7])
    lower_box = np.array([P1, P5, P8, P4])

    boxes = [left_box, upper_box, right_box, lower_box]

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([tuple(p) for p in box], fill=fill_color, width=0, outline=fill_color)
    return img


def draw_snake(img, width=5, n_lines_max=5, border_padding=10):
    """
    Draw a snaking line on the image, the line will change directions maximum n_lines_max times
    """
    fill_color = 'white'
    resolution = np.array([img.width, img.height])

    img_snake = Image.new('L', (resolution[0], resolution[1]), color=(0))
    draw = ImageDraw.Draw(img_snake)

    border_padding = 10
    min_length = 0.2 * min(resolution)  # 20% of the smaller dimension of the image
    margin = min_length / 2
    x, y = np.random.uniform(border_padding, resolution[0] - border_padding), np.random.uniform(border_padding, resolution[1] - border_padding)
    current_angle = np.random.uniform(0, 360)

    # Adjust initial direction if it points out of the image
    end_x = x + min_length * np.cos(np.radians(current_angle))
    end_y = y + min_length * np.sin(np.radians(current_angle))
    while not (border_padding <= end_x < (resolution[0] - border_padding) and border_padding <= end_y < (resolution[1] - border_padding)):
        current_angle = np.random.uniform(0, 360)
        end_x = x + min_length * np.cos(np.radians(current_angle))
        end_y = y + min_length * np.sin(np.radians(current_angle))
    drawn = False
    n_lines = 0
    while True:
        length = np.random.uniform(min_length, min_length * 4)  # 20% - 30% of image length

        valid_move = False
        attempts = 0
        if n_lines == 0:
            new_angle = current_angle

        while not valid_move and attempts < 20:  # Limit the number of attempts to find a valid direction
            end_x = x + length * np.cos(np.radians(new_angle))
            end_y = y + length * np.sin(np.radians(new_angle))

            # Check if the end point is inside the image boundary
            if border_padding <= end_x < (resolution[0] - border_padding) and border_padding <= end_y < (resolution[1] - border_padding):
                valid_move = True
                current_angle = new_angle
            else:
                new_angle = current_angle + np.random.uniform(-90, 90)  # Change direction
                attempts += 1

        if not valid_move:
            break  # Stop drawing if a valid move isn't found after several attempts

        draw.line([x, y, end_x, end_y], fill=fill_color, width=width)
        drawn = True
        n_lines = n_lines + 1

        if n_lines >= n_lines_max:
            break
        # Update position and angle for the next segment
        backward_vec = np.array([x, y]) - np.array([end_x, end_y])
        backward_vec_norm = backward_vec / np.linalg.norm(backward_vec)
        # print(np.linalg.norm(backward_vec_norm))

        start_point = np.array([end_x, end_y]) + width / 8 * backward_vec_norm
        x, y = start_point[0], start_point[1]
        new_angle = current_angle + np.random.choice([-30, 30])  # Change direction for the next segment
    if drawn == False:
        draw = draw_snake(img, width, n_lines_max)

    img = ImageChops.lighter(img, img_snake)
    return img


def draw_high_variation_img(dimension=np.array([0.9, 0.6]),
                            resolution=np.array([640, 480]),
                            draw_scaling=2.0,
                            max_beading_ratio=0.5,
                            n_lines=2,
                            n_snakes=1,
                            n_rect=1,
                            n_arc=1,
                            height=0.02,
                            sym="all",
                            eng_beading=None,
                            min_length_scale=0):

    draw_resolution = resolution * draw_scaling

    if isinstance(height, list): height_sample = np.random.uniform(0.005, 0.02, 1)
    else: height_sample = height

    # Create white image
    empty_img = True
    beading_ratio_check = True
    while empty_img | beading_ratio_check:

        img = Image.new('L', (int(draw_resolution[0]), int(draw_resolution[1])), color=(0))
        A = draw_resolution[0] * draw_resolution[1]
        pixel_per_mm = np.max(draw_resolution / (dimension * 1000))

        if isinstance(n_lines, list): n_lines_plot = np.random.randint(n_lines[0], n_lines[1] + 1)
        else: n_lines_plot = n_lines
        min_l = 450
        width = np.random.uniform(30 * pixel_per_mm, 80 * pixel_per_mm, n_lines_plot)
        for i in range(n_lines_plot):
            cond = True
            while cond:
                P1x = np.random.uniform(pixel_per_mm * 55, draw_resolution[0] - pixel_per_mm * 55, 1)
                P1y = np.random.uniform(pixel_per_mm * 55, draw_resolution[1] - pixel_per_mm * 55, 1)
                P2x = np.random.uniform(pixel_per_mm * 55, draw_resolution[0] - pixel_per_mm * 55, 1)
                P2y = np.random.uniform(pixel_per_mm * 55, draw_resolution[1] - pixel_per_mm * 55, 1)
                l = np.sqrt((P2x - P1x)**2 + (P2y - P1y)**2)
                if l > min_l * pixel_per_mm:
                    cond = False
            img = draw_line_P1P2(img, P1=[P1x, P1y], P2=[P2x, P2y], width=int(width[i]))

        # draw 0 - 2 snake lines
        if isinstance(n_snakes, list): n_snakes_plot = np.random.randint(n_snakes[0], n_snakes[1] + 1)
        else: n_snakes_plot = n_snakes
        width = np.random.uniform(50 * pixel_per_mm, 70 * pixel_per_mm)
        for i in range(n_snakes_plot):
            img = draw_snake(img, width=int(width), n_lines_max=5)

        # draw 0 - 2 Quads
        if isinstance(n_rect, list): n_rect_plot = np.random.randint(n_rect[0], n_rect[1] + 1)
        else: n_rect_plot = n_rect
        for i in range(n_rect_plot):
            x_mid = np.random.uniform(pixel_per_mm * 200, draw_resolution[0] - pixel_per_mm * 200)
            y_mid = np.random.uniform(pixel_per_mm * 200, draw_resolution[0] - pixel_per_mm * 200)
            length_x = np.random.uniform(pixel_per_mm * 150, draw_resolution[0] - pixel_per_mm * 110)
            length_y = np.random.uniform(pixel_per_mm * 150, draw_resolution[1] - pixel_per_mm * 110)
            while length_x * length_y > 0.3 * A:
                length_x = np.random.uniform(pixel_per_mm * 150, draw_resolution[0] - pixel_per_mm * 110)
                length_y = np.random.uniform(pixel_per_mm * 150, draw_resolution[1] - pixel_per_mm * 110)

            angle = np.random.uniform(0.0, 180)
            choice = np.random.uniform(0.0, 1.0)
            if choice > 0.5: margin = np.random.uniform(0.2, 0.3)
            else: margin = None
            img = draw_rectangle_hole(img, length_x, length_y, x_mid, y_mid, margin=margin, angle=angle)

        # draw 0 - 2 Ellipse
        if isinstance(n_arc, list): n_ellipse_plot = np.random.randint(n_arc[0], n_arc[1] + 1)
        else: n_ellipse_plot = n_arc
        for i in range(n_ellipse_plot):
            x_mid = np.random.uniform(0.0, draw_resolution[0])
            y_mid = np.random.uniform(0.0, draw_resolution[1])
            length_x = np.random.uniform(pixel_per_mm * 150, draw_resolution[0] - pixel_per_mm * 110)
            length_y = np.random.uniform(pixel_per_mm * 150, draw_resolution[1] - pixel_per_mm * 110)
            angle = np.random.uniform(0.0, 180)
            #short_side = np.min([length_x, length_y])
            #width = np.random.uniform(short_side * 0.25, short_side * 0.35)
            width = np.random.uniform(40 * pixel_per_mm, 100 * pixel_per_mm)
            choice = np.random.uniform(0.0, 1.0)
            if choice > 0.5: end_angle = np.random.uniform(120.0, 360)
            else: end_angle = 360
            img = draw_arc(img, length_x, length_y, x_mid=x_mid, y_mid=y_mid, width=int(width), end_angle=end_angle, angle=angle)

        switch = np.random.uniform(0, 1)

        if sym == "all":
            bounds = [0.33, 0.66, 1.0]
        elif sym == "partial":
            bounds = [0.25, 0.5, 0.75]
        elif sym == "no":
            bounds = [0.0, 0.0, 0.0]

        if switch > 0.0 and switch < bounds[0]:
            img = mirror_quater(img)
        elif switch > bounds[0] and switch < bounds[1]:
            img = mirror_lr(img)
        elif switch > bounds[1] and switch < bounds[2]:
            img = mirror_ud(img)

        img = draw_bounding_box_2(img, width=pixel_per_mm * 50)

        height_map = np.array(img,dtype=float)
        height_map/=255 # norm to [0, 1]
        height_map*=0.02
        # nan assert
        assert np.isnan(height_map).sum() == 0, "Height map contains NaN values"
        height_map = postprocess_plate(height_map,dimensions=(dimension[1],dimension[0]),
                                       min_length_scale=min_length_scale,eng_beading=eng_beading)
        height_map/=0.02
        height_map*=255
        img = Image.fromarray(height_map)

        img = img.resize((int(resolution[0]), int(resolution[1])), Image.BOX)


        height_mat = get_height_mat(img, height=height_sample)
        if np.max(height_mat) > 0.0:
            empty_img = False

            # Calculate beading ratio
            beading_ratio = np.sum(height_mat) / (resolution[0] * resolution[1] * height_sample)
            #continue as long the beading_ratio is over max_beading_ratio
            beading_ratio_check = beading_ratio > max_beading_ratio

    return height_mat


class BeadingTransition():
    """
    TODO Give short description of purpose of class
    """

    def __init__(self, h_bead, r_f, r_h, alpha_F) -> None:
        self.h_bead = h_bead
        self.r_f = r_f
        self.r_h = r_h
        self.alpha_F = alpha_F

        self.P0, self.P1, self.P2, self.P3 = self.calculate_design_points()

        self.xi_upper_bounds = np.array([self.P1[0], self.P2[0], self.P3[0]])

    def calculate_design_points(self):
        P0 = np.array([0.0, 0.0])
        P1 = np.array([self.r_f * np.sin(self.alpha_F), self.r_f - self.r_f * np.cos(self.alpha_F)])
        P2_y = self.h_bead - self.r_h + self.r_h * np.cos(self.alpha_F)
        P2_x = P1[0] + (P2_y - P1[1]) / np.tan(self.alpha_F)
        P2 = np.array([P2_x, P2_y])
        P3 = np.array([P2_x + self.r_h * np.sin(self.alpha_F), self.h_bead])

        return P0, P1, P2, P3

    def plot_beat(self, offset=None, ax=None):

        if ax is None:
            fig, ax = plt.subplots()
        xi = np.linspace(0.0, self.P3[0], 1000)
        h = self.get_h(xi)

        if offset is not None:
            xi = xi + offset

        ax.plot(xi, h)
        ax.set_xlabel("xi")
        ax.set_ylabel("h")
        ax.axis('equal')

        return ax

    def get_h(self, xis):
        h = np.zeros(len(xis))

        for i, xi in enumerate(xis):
            if xi < self.xi_upper_bounds[0]:
                if xi < 0.0:
                    xi = 0.0
                h[i] = self.r_f - np.sqrt(self.r_f**2 - xi**2)
            elif (xi >= self.xi_upper_bounds[0]) & (xi < self.xi_upper_bounds[1]):
                h[i] = xi * np.tan(self.alpha_F) + self.P1[1] - self.P1[0] * np.tan(self.alpha_F)
            elif (xi >= self.xi_upper_bounds[1]) & (xi <= self.xi_upper_bounds[2]):
                h[i] = self.h_bead - self.r_h + np.sqrt(self.r_h**2 - (self.P3[0] - xi)**2)

        return h

    def round_float_to_next_even(self, num):
        rounded_num = int(np.ceil(num))
        if rounded_num % 2 == 0:
            return rounded_num
        else:
            return rounded_num + 1

    def get_hat_mat(self, dimension, resolution, scaling=1.0):

        dx = dimension[0] / (resolution[0] - 1) / scaling
        dy = dimension[1] / (resolution[1] - 1) / scaling

        w_hat = 2 * self.P3[0]
        n_elem_x = self.round_float_to_next_even(w_hat / dx)
        n_elem_y = self.round_float_to_next_even(w_hat / dy)

        l_x = n_elem_x * dx
        l_y = n_elem_y * dy

        x_coord = np.linspace(-l_x / 2, l_x / 2, n_elem_x + 1).reshape(1, -1)
        y_coord = np.linspace(-l_y / 2, l_y / 2, n_elem_y + 1).reshape(-1, 1)

        x_coord_mat = np.repeat(x_coord, n_elem_y + 1, axis=0)
        y_coord_mat = np.repeat(y_coord, n_elem_x + 1, axis=1)

        dist_mat = -np.sqrt(x_coord_mat**2 + y_coord_mat**2) + w_hat / 2

        dist_mat_flat = dist_mat.flatten()

        height = self.get_h(dist_mat_flat)
        height_mat = height.reshape(dist_mat.shape)

        return height_mat


class ParametricBeading:
    def __init__(self, resolution, dimension, eng_beading, min_length_scale = 0.0, draw_scaling = 2) -> None:
        self.resolution = resolution
        self.draw_resolution = resolution * draw_scaling
        self.dimension = dimension
        self.pixel_per_mm = np.max(self.draw_resolution / (dimension * 1000))
        self.eng_beading = eng_beading
        self.min_length_scale = min_length_scale

    def scale_line_para(self, theta):
        x_inter = [self.pixel_per_mm * 55, self.draw_resolution[0] - self.pixel_per_mm * 55]
        y_inter = [self.pixel_per_mm * 55, self.draw_resolution[1] - self.pixel_per_mm * 55]
        w_interval = [30 * self.pixel_per_mm, 80 * self.pixel_per_mm]
        para_interval = np.array([x_inter, y_inter, x_inter, y_inter, w_interval])
        p = para_interval[:,0] + (para_interval[:,1] - para_interval[:,0]) * theta
        return p

    def scale_arc_para(self, theta):
        lx_inter = [self.pixel_per_mm * 150, self.draw_resolution[0] - self.pixel_per_mm * 110]
        ly_inter = [self.pixel_per_mm * 150, self.draw_resolution[1] - self.pixel_per_mm * 110]
        xmid_interval = [0.0, self.draw_resolution[0]]
        ymid_interval = [0.0, self.draw_resolution[1]]
        width = [40 * self.pixel_per_mm, 100 * self.pixel_per_mm]
        end_angel = [120.0, 360]
        angle = [0.0, 180]
        switch = [0.0, 1.0]
        para_interval = np.array([lx_inter, ly_inter, xmid_interval, ymid_interval, width, end_angel, angle, switch])
        p = para_interval[:,0] + (para_interval[:,1] - para_interval[:,0]) * theta
        return p
    
    def scale_quad_para(self, theta):
        x_mid = [self.pixel_per_mm * 200, self.draw_resolution[0] - self.pixel_per_mm * 200]
        y_mid = [self.pixel_per_mm * 200, self.draw_resolution[0] - self.pixel_per_mm * 200]
        length_x = [self.pixel_per_mm * 150, self.draw_resolution[0] - self.pixel_per_mm * 110]
        length_y = [self.pixel_per_mm * 150, self.draw_resolution[1] - self.pixel_per_mm * 110]
        angle = [0.0, 180]
        margin = [0.2, 0.3]
        switch = [0.0, 1.0]
        para_interval = np.array([length_x, length_y, x_mid, y_mid, margin, angle, switch])
        p = para_interval[:,0] + (para_interval[:,1] - para_interval[:,0]) * theta
        return p
    
    def random_uniform_integer(self, r, random_variable):
        if not (0 <= random_variable <= 1):
            raise ValueError("random_variable must be between 0 and 1.")
        return int(random_variable * (r+1))

    def draw(self, theta, max_n_lines = 1, max_n_arcs = 0, max_n_quads = 0):

        n_line_para = 5
        n_quad_para = 7
        n_arc_para = 8

        if len(theta) != (max_n_lines * n_line_para + max_n_arcs * n_arc_para + max_n_quads * n_quad_para + 3):
            raise ValueError("The number of parameter does match the number of geometries")
        
        beadings = np.zeros((theta.shape[1], 1, self.resolution[1], self.resolution[0]))
  
        # Last two parameters are switch parameters
        for n in range(theta.shape[1]):
            img = Image.new('L', (int(self.draw_resolution[0]), int(self.draw_resolution[1])), color=(0))

            n_lines = self.random_uniform_integer(max_n_lines, theta[-1,n])
            for i in range(n_lines):    
                p = self.scale_line_para(theta[n_line_para*i:n_line_para*(i+1),n])
                img = draw_line_P1P2(img, P1=[p[0], p[1]], P2=[p[2], p[3]], width=int(p[4]))

            n_arcs = self.random_uniform_integer(max_n_arcs, theta[-2,n])
            for i in range(n_arcs):
                idx_start = n_line_para * max_n_lines
                p = self.scale_arc_para(theta[idx_start + n_arc_para*i:idx_start + n_arc_para*(i+1),n])
                if p[7] > 0.5: end_angle = p[5]
                else: end_angle = 360
                img = draw_arc(img, p[0], p[1], x_mid=p[2], y_mid=p[3], width=int(p[4]), end_angle=end_angle, angle=p[6])

            n_quads = self.random_uniform_integer(max_n_quads, theta[-3,n])
            for i in range(n_quads):
                idx_start = n_line_para * max_n_lines + n_arc_para * max_n_arcs
                p = self.scale_quad_para(theta[idx_start + n_quad_para*i:idx_start + n_quad_para*(i+1),n])
                if p[6] > 0.5: margin = p[4]
                else: margin = None
                img = draw_rectangle_hole(img, p[0], p[1], p[2], p[3], margin = margin, angle = p[5])

            img = draw_bounding_box_2(img, width=self.pixel_per_mm * 50)

            height_map = np.array(img,dtype=float)
            height_map/=255 # norm to [0, 1]
            height_map*=0.02
            # nan assert
            assert np.isnan(height_map).sum() == 0, "Height map contains NaN values"
            height_map = postprocess_plate(height_map,dimensions=(self.dimension[1],self.dimension[0]),
                                        min_length_scale=self.min_length_scale,eng_beading=self.eng_beading)
            height_map/=0.02
            height_map*=255
            img = Image.fromarray(height_map)
            
            
            img = img.resize((int(self.resolution[0]), int(self.resolution[1])), Image.BOX)

            beading = get_height_mat(img)
            beadings[n,0] = beading

        return beadings

def postprocess_plate(height_map,dimensions=(0.6,0.9),
            min_length_scale = DEFAULT_MIN_LENGTH_SCALE,
            eng_beading = BeadingTransition(h_bead=0.02, r_f=0.0095, r_h=0.0095, alpha_F=70 * np.pi / 180)
                      ):
    hat_mat = eng_beading.get_hat_mat(dimensions, height_map.shape)
    height_map = apply_length_scale_constraint(height_map,min_length_scale,dimensions,hat_mat.max())
    height_map = apply_engineering_blur(height_map, hat_mat)
    return height_map

def postprocess(height_map, times=2):
    for _ in range(times):
        height_map = np.array([postprocess_plate(hm) for hm in height_map])
    height_map = torch.from_numpy(height_map).unsqueeze(1)
    return height_map


from concurrent.futures import ProcessPoolExecutor
from functools import partial


def identity(x):
    return x

def compute_batch(drawing_fn, transform, args):
    indices, seed = args
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.stack([
        transform(torch.from_numpy(drawing_fn()).unsqueeze(0)) for _ in indices
    ], dim=0)


class PlateDataset(Dataset):
    def __init__(self, drawing_fn, transform=None, size=10000, num_threads=50, batch_size=2000):
        self.n_samples = size
        self.drawing_fn = drawing_fn
        self.transform = transform or identity
        sample_shape = self.transform(torch.from_numpy(drawing_fn()).unsqueeze(0)).shape
        self.data = torch.empty((self.n_samples, *sample_shape), dtype=torch.float)
        n_batches = (self.n_samples + batch_size - 1) // batch_size
        batch_indices = [range(i * batch_size, min((i + 1) * batch_size, self.n_samples))
                         for i in range(n_batches)]
        # Generate per-batch seeds
        seed_sequence = np.random.SeedSequence()
        seeds = seed_sequence.spawn(len(batch_indices))
        seed_ints = [s.generate_state(1)[0] for s in seeds]

        # Combine indices and seeds for executor.map
        batch_args = list(zip(batch_indices, seed_ints))

        p_compute_batch = partial(compute_batch, self.drawing_fn, self.transform)

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            batches = list(executor.map(p_compute_batch, batch_args))
        start = 0
        for batch in batches:
            bs = batch.shape[0]
            self.data[start:start+bs] = batch
            start += bs

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], torch.zeros(1)


# V5000PlateDataset = PlateDataset(draw_simple_img)

# Definition of the D50000 plate dataset for generative model training

def _StandardPlateDataset(min_length_scale=0.01,**kwargs):
    eng_beading = BeadingTransition(h_bead=0.02, r_f=0.0095, r_h=0.0095, alpha_F=70 * np.pi / 180)
    draw_D50000_img = lambda: draw_high_variation_img(dimension=np.array([0.9, 0.6]),
                                                    resolution=np.array([181, 121]),
                                                    height=0.02,
                                                    n_lines=[1, 2],
                                                    n_snakes=[0, 1],
                                                    n_rect=[0, 2],
                                                    n_arc=[0, 2],
                                                    sym="partial",
                                                    eng_beading=eng_beading,
                                                    max_beading_ratio=0.5)

    return PlateDataset(draw_D50000_img, **kwargs)

def standard_draw_img(eng_beading, min_length_scale):
    return draw_high_variation_img(
        dimension=np.array([0.9, 0.6]),
        resolution=np.array([181, 121]),
        height=0.02,
        n_lines=[1, 2],
        n_snakes=[0, 1],
        n_rect=[0, 2],
        n_arc=[0, 2],
        sym="partial",
        eng_beading=eng_beading,
        max_beading_ratio=0.5,
        min_length_scale=min_length_scale)


def StandardPlateDataset(min_length_scale=0.025, **kwargs):
    print(f"StandardPlateDataset, min_length_scale={min_length_scale}")
    eng_beading = BeadingTransition(h_bead=0.02, r_f=0.0095, r_h=0.0095, alpha_F=70 * np.pi / 180)
    draw_img = partial(standard_draw_img, eng_beading, min_length_scale)
    return PlateDataset(draw_img, **kwargs)
    

