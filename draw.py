from p5 import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Global variables
world_size = 255
# world = np.array([[120 for _ in range(world_size + 1)] for _ in range(world_size + 1)], dtype=np.uint8)
world = np.full(shape=(world_size, world_size), fill_value=255, dtype=np.uint8)
brush_stroke = 3

def setup():
    size(world_size, world_size)
    background(255, 255, 255)

def circle(arry, y, x):
    x_start = max(0, x - brush_stroke)
    x_end = min(255, x + brush_stroke)
    y_start = max(0, y - brush_stroke)
    y_end = min(255, y + brush_stroke)
    arry[x_start:x_end, y_start:y_end] = 0

def draw():
    if mouse_is_pressed:
        global world
        x, y = floor(mouse_x), floor(mouse_y)
        circle(world, x, y)
        # world[x][y] = 255
        no_stroke()
        fill(0, 0, 0)
        rect((x, y), brush_stroke * 2, brush_stroke * 2)
    elif key_is_pressed:
        no_loop()
        # img = np.full((256, 256), 255, dtype=np.uint8)
                    # print(x)
        # img = np.zeros((256, 256))
        # img = np.zeros((100,100), dtype=np.uint8)
        # img = np.full(shape=(100, 100), fill_value=255, dtype=np.uint8)
        # for r in img:
        #     for i in r:
        #         print(i)
        # print(img)
        # print(img.shape)
        # input()
        # for i in world:
        #     for a in i:
        #         if a != 255: print(a)
        # print(world)
        img = Image.fromarray(world)
        # img.show()
        img.save("pre_sample.png", "png")
        img.resize((28, 28), Image.ANTIALIAS).save("test.png", "png")
        print(np.asarray(img).shape)
        exit()

        
if __name__ == '__main__':
    run(sketch_setup=setup, sketch_draw=draw, frame_rate=120)
