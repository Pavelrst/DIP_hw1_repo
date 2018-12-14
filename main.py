import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
import scipy.io


class Trajectories:
    def __init__(self):
        mat = scipy.io.loadmat('100_motion_paths.mat')
        self.x_vals = mat['X']
        self.y_vals = mat['Y']

    def get_trajectory(self,index):
        # This method returns an x,y points for
        # a trajecory of a given index.
        x = self.x_vals[index]
        y = self.y_vals[index]
        return x,y

    def plot_trajectory(self,index):
        x,y = self.get_trajectory(index)
        fig, ax1 = plt.subplots()
        ax1.plot(x, y)
        fig.tight_layout()
        plt.show()

    def generate_kernel(self,index, show = False):
        x, y = self.get_trajectory(index)
        kernel_size = 9
        center_shift = (kernel_size-1)/2
        kernel = np.zeros((kernel_size,kernel_size),int)
        zoom_factor = 1

        print("len x = ",len(x))
        for k in range(len(x)):
            kernel_row = int(round(center_shift+x[k]*zoom_factor))
            kernel_col = int(round(center_shift-y[k]*zoom_factor))
            if kernel_col<kernel_size and kernel_row<kernel_size:
                kernel[kernel_col, kernel_row] = kernel[kernel_col, kernel_row] + 1

        if show == True:
            #img = Image.fromarray(kernel)
            #img.show()
            plt.imshow(kernel, cmap='gray')
            plt.show()

        return kernel

class Image:
    def __init__(self):
        self.image = mpimg.imread('DIPSourceHW1.jpg')
        self.image_arr = np.array(self.image)[:, :, 0]

    def get_2d_array(self):
        return self.image_arr

    def apply_filter(self,filter,show = False):
        filtered = scipy.signal.convolve2d(self.image_arr,filter)

        if show == True:
            #img = Image.fromarray(filtered)
            #img.show()
            plt.imshow(filtered, cmap='gray')
            plt.show()
        return filtered

def main():
    my_T = Trajectories()
    my_Im = Image()

    n = 5
    for i in range(n):
        x, y = my_T.get_trajectory(i)
        kernel = my_T.generate_kernel(i)
        filtered = my_Im.apply_filter(kernel)
        plt.subplot(3, n, i+1)
        plt.plot(x,y)
        plt.subplot(3, n, n+i+1)
        plt.imshow(kernel, cmap='gray')
        plt.subplot(3, n, 2*n+i+1)
        plt.imshow(filtered, cmap='gray')

    plt.show()

if __name__ == '__main__':
    main()