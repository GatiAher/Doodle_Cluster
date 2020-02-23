from quickdraw import QuickDrawData
from PIL import Image
import numpy as np

def get_array_numpy_images():
    """returns labels, ndarry (shape (~, 28, 28))"""

    # max_drawings -- maximum number of drawings to be loaded into memory,
    # defaults to 1000 (3 minutes)
    qd = QuickDrawData(max_drawings=1000)

    # LIST OBJECT
    drawing_names = qd.drawing_names

    # LABELS
    list_of_labels = []

    # just append your arrays to a Python list and convert it at the end;
    # the result is simpler and faster
    list_of_np = []

    for label in drawing_names:

        # QuickDrawDataGroup Object
        qd_data_group = qd.get_drawing_group(label)

        # QuickDrawing Object
        for drawing in qd_data_group.drawings:

            list_of_labels.append(label)

            # PIL IMAGE
            PIL_im = drawing.image.convert('L').resize((28, 28), Image.ANTIALIAS)
            nump_im = np.array(PIL_im)
            list_of_np.append(nump_im)

    # convert to ndarray
    array_of_np = np.asarray(list_of_np)
    array_of_labels = np.asarray(list_of_labels)
    print("IMAGES SHAPE ", array_of_np.shape)
    print("LABELS SHAPE ", array_of_labels.shape)

    np.save("images-new.npy", array_of_np)
    np.save("labels-new.npy", array_of_labels)

if __name__== "__main__":
    get_array_numpy_images()
