import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def show_slices(slices, cmap):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap=cmap, origin="lower")

def display(img_data, block=True, title=None, cmap='gray'):
    if len(img_data.shape) > 3 : img_data = np.squeeze(img_data)
    n_i, n_j, n_k = img_data.shape
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2
    print('Image center: ', center_i, center_j, center_k)
    center_vox_value = img_data[center_i, center_j, center_k]
    print('Image center value: ', center_vox_value)

    slice_0 = img_data[center_i, :, :]
    slice_1 = img_data[:, center_j, :]
    slice_2 = img_data[:, :, center_k]

    show_slices([slice_0, slice_1, slice_2], cmap=cmap)

    plt.suptitle("Center slices for image")
    if title:
        plt.suptitle(title)
    plt.show(block = block)
    return plt


def display_4D(img_data, block=False, title=None, cmap='gray', fps=30):
    if len(img_data.shape) > 4: img_data = np.squeeze(img_data)
    n_z = img_data.shape[2]
    n_t = img_data.shape[3]
    center_z = (n_z - 1) // 2
    z_slice_data = img_data[:, :, center_z, :]
    first_time_point = z_slice_data[..., 0]
    fig = plt.figure()
    im = plt.imshow(first_time_point.T, cmap=cmap, origin="lower")

    def animate(i):
        if i % fps == 0:
            print('.', end='')

        im.set_array(z_slice_data[..., i].T)
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=n_t,
        interval=1000 / fps,  # in ms
    )
    plt.show(block=block)

    return plt
