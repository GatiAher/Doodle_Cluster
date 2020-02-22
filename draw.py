from p5 import *
import random
from random import uniform
import math
from datetime import datetime

# Global variables
world_size = 999
world = [[0 for _ in range(world_size + 1)] for _ in range(world_size + 1)]

def setup():
    size(world_size, world_size)
    background(255, 255, 255)

def draw():
    if mouse_is_pressed:
        global world
        x, y = floor(mouse_x), floor(mouse_y)
        if (world[x][y] == 0 and
            (x >= 0 and x <= world_size) and
            (y >= 0 and y <= world_size)):
            world[x][y] = 1
            no_stroke()
            fill(0, 0, 0)
            rect((x, y), 10, 10)
        
if __name__ == '__main__':
    run(sketch_setup=setup, sketch_draw=draw, frame_rate=60)
