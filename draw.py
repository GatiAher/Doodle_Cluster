from p5 import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Global variables
world_size = 255
world = np.full(shape=(world_size, world_size), fill_value=255, dtype=np.uint8)
brush_stroke = 3

def setup():
    # Create world with blank background
    size(world_size, world_size)
    fill(255, 255, 255)
    rect((0, 0), world_size, world_size)
    # Set rectangle mode center so we draw rectangles around the mouse
    rect_mode(mode='CENTER')

def circle(arry, y, x, value=0):
    """
    Takes in an numpy array and mouse coordinates, and fills in a square of
    values around the coordinates.

    Parameters:
        arry: Numpy array of (world_size, world_size)
        x: X coordinate of mouse
        y: Y coordinate of mouse
        value: Value to fill the array with
    """
    x_start = max(0, x - brush_stroke)
    x_end = min(world_size, x + brush_stroke)
    y_start = max(0, y - brush_stroke)
    y_end = min(world_size, y + brush_stroke)
    arry[x_start:x_end, y_start:y_end] = value

def process_image(arry):
    """
    Takes in an image and crops to the smallest possible square containing all
    points.

    Parameters:
        arry: Square 2D numpy array containg 0s where a point has been drawn

    Returns:
        Array: Passed numpy array cropped down to smallest possible square
        containing all points
    """
    # Get all points where we've drawn with argwhere()
    locs = np.argwhere(arry == 0)
    locs = np.transpose(locs)
    xs, ys = locs[0], locs[1]

    # Get the min and max x and y values
    # Allows us to draw a box around the data to crop to
    x_start, x_end = min(xs), max(xs)
    y_start, y_end = min(ys), max(ys)
    # Cut out the doodle
    s = arry[x_start:x_end + 1, y_start:y_end + 1]

    # Calculate and compare x and y range to make our box a square
    x_range = x_end - x_start
    y_range = y_end - y_start
    final_size = max(x_range, y_range) + 1
    difference = abs(x_range - y_range)
    delta = difference // 2
    # Handles if the difference in range is odd
    handle_odd = difference % 2

    # Create a new array to hold the slice
    ret = np.full(shape=(final_size, final_size), fill_value=255, dtype=np.uint8)
    # Calculate location in the new array to add the slice to
    if x_range > y_range:
        y_start, y_end = delta + handle_odd, -delta
        x_start, x_end = 0, None
    elif y_range > x_range:
        x_start, x_end = delta + handle_odd, -delta
        y_start, y_end = 0, None
    # Insert the slice
    ret[x_start:x_end, y_start:y_end] = s
    return ret

def draw():
    if mouse_is_pressed:
        global world
        x, y = floor(mouse_x), floor(mouse_y)
        # Add data to our world array...
        circle(world, x, y)
        # ... and draw a rectangle to the screen around the pointer
        no_stroke()
        fill(0, 0, 0)
        rect((x, y), brush_stroke * 2, brush_stroke * 2)
    elif key_is_pressed:
        # If key is pressed, stop and save the image
        # Stop processing draw() loop
        no_loop()
        # Convert to PIL image
        img = process_image(world)
        Image.fromarray(world).save("pre_scale.png", "png")
        img = Image.fromarray(img)
        img.save("post_scale.png", "png")
        # Scale image down to required size and save
        img.resize((28, 28), Image.ANTIALIAS).save("final.png", "png")
        exit()

        
if __name__ == '__main__':
    run(sketch_setup=setup, sketch_draw=draw, frame_rate=120)
