import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import signal
from scipy import fftpack
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
        # This method generates a PSF according
        # to given index of trajectory.
        # Notice: PSF == kernel
        x, y = self.get_trajectory(index)
        kernel_size = 13 # set to 9-15
        center_shift = (kernel_size-1)/2
        kernel = np.zeros((kernel_size,kernel_size),int)
        zoom_factor = 1 # set to 1

        for k in range(len(x)):
            kernel_row = int(round(center_shift+x[k]*zoom_factor))
            kernel_col = int(round(center_shift-y[k]*zoom_factor))
            if kernel_col<kernel_size and kernel_row<kernel_size:
                kernel[kernel_col, kernel_row] = kernel[kernel_col, kernel_row] + 1

        if show == True:
            plt.imshow(kernel, cmap='gray')
            plt.show()
        return kernel

class Blurr_Applier:
    def __init__(self,num):
        self.num_of_traj = num
        self.trajectories = Trajectories()
        self.psfs = list()
        self.blurred_images = list()

    def generate_PSFs(self):
        for i in range(self.num_of_traj):
            self.psfs.append(self.trajectories.generate_kernel(i))

    def apply_blurr(self):
        my_Im = Image_cl()
        for i in range(self.num_of_traj):
            self.blurred_images.append(my_Im.apply_filter(self.psfs[i]))

    def plot_batches_of_5(self):
        for batch in range(0,20,1):
            n = 5 # num of images in batch
            for t in range(5):
                x,y = self.trajectories.get_trajectory(t+batch*5)
                plt.subplot(3, 5, t+1)
                plt.plot(x,y)
                plt.subplot(3, 5, n+t+1)
                plt.imshow(self.psfs[t+batch*5], cmap='gray')
                plt.subplot(3, 5, 2*n+t+1)
                plt.imshow(self.blurred_images[t+batch*5], cmap='gray')
            plt.show()

    def plot_batches_of_10(self):
        for batch in range(0,10,1):
            n = 10 # num of images in batch
            for t in range(10):
                x,y = self.trajectories.get_trajectory(t+batch*5)
                plt.subplot(3, 10, t+1)
                plt.plot(x,y)
                plt.subplot(3, 10, n+t+1)
                plt.imshow(self.psfs[t+batch*10], cmap='gray')
                plt.subplot(3, 10, 2*n+t+1)
                plt.imshow(self.blurred_images[t+batch*10], cmap='gray')
            plt.show()

    def save_plot_to_image(self,path):
            for t in range(self.num_of_traj):
                x,y = self.trajectories.get_trajectory(t)
                plt.subplot2grid((2, 3), (0, 2))
                plt.plot(x,y)
                plt.subplot2grid((2, 3), (1, 2))
                plt.imshow(self.psfs[t], cmap='gray')
                plt.subplot2grid((2, 3), (0, 0),colspan=2, rowspan=2)
                plt.imshow(self.blurred_images[t], cmap='gray')
                plt.tight_layout()
                full_path = path + str(t) + '.png'
                plt.savefig(full_path)
                #plt.show()
                plt.clf()


    def get_blurred_images(self):
        return self.blurred_images


class Blurr_Fixer:
    def __init__(self, blurred_images, p=1,ifft_scale=1000, original_size=256, margin=0):
        self.blurred_images = blurred_images
        self.F_images = list()
        self.p = p
        self.ifft_scale = ifft_scale
        for img in blurred_images:
            self.F_images.append(fftpack.fftshift(fftpack.fftn(img)))
        self.fixed = []
        self.margin = margin
        self.original_size = original_size

    def calc_weights_denom(self):
        weights_denom = np.zeros(self.F_images[0].shape)
        for mat in self.F_images:
            weights_denom = weights_denom + np.power(np.abs(mat), self.p)
        return weights_denom

    def fix_blurr(self):
        denom = self.calc_weights_denom()
        accumulator = np.zeros(self.F_images[0].shape)
        for F in self.F_images:
            curr_weight = np.divide(np.power(np.abs(F), self.p), denom)
            accumulator = accumulator + np.multiply(F, curr_weight)
        fixed = fftpack.ifft2(fftpack.ifftshift(accumulator)).real
        fixed = np.divide(fixed,self.ifft_scale)
        # Crop
        size = self.original_size
        margin = self.margin
        self.fixed = fixed[margin:margin + size, margin:margin + size]
        return self.fixed

    def get_FT(self):
        return self.F_image

    def show_Fixed(self):
        original = mpimg.imread('DIPSourceHW1.jpg')
        p1 = plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        p1.set_title("Original Image")
        p2 = plt.subplot(1, 2, 2)
        plt.imshow(self.fixed, cmap='gray')
        p2.set_title("Fourier Burst Accumulation")
        plt.show()

class Img_Avg():
    def __init__(self, blurred_images):
        self.blurred_images = blurred_images
        self.num_img = len(blurred_images)

    def get_avg(self):
        avg_img = np.zeros(self.blurred_images[0].shape)
        for img in self.blurred_images:
            avg_img = avg_img + np.divide(img,self.num_img)
        return avg_img

    def get_sharp_avg(self):
        avg = self.get_avg()
        kernel = [[0,-2,0],[-2,9,-2],[0,-2,0]]
        sharpen = scipy.signal.convolve2d(avg, kernel)
        return sharpen

class Image_cl:
    def __init__(self):
        self.image = mpimg.imread('DIPSourceHW1.jpg')
        self.image_arr = np.array(self.image)[:, :, 0]

    def get_2d_array(self):
        return self.image_arr

    def get_image(self):
        return self.image

    def apply_filter(self,filter,show = False):
        filtered = scipy.signal.convolve2d(self.image_arr,filter)

        if show == True:
            #img = Image.fromarray(filtered)
            #img.show()
            plt.imshow(filtered, cmap='gray')
            plt.show()
        return filtered

class PSNR_calculator:
    def __init__(self,original_image,fixed_image):
        self.orig = original_image[:,:,0]
        self.fixed = np.array(fixed_image)

    def evaluate_PSNR(self):
        MSE = np.mean(np.power(np.abs(np.subtract(self.fixed,self.orig)),2))
        MAX = 255
        return 10*np.log(np.power(MAX,2)/MSE)



def main():
    my_Im = Image_cl()

    my_applier = Blurr_Applier(100)
    my_applier.generate_PSFs()
    my_applier.apply_blurr()
    print("Plot Trajectories, PSFs and Blurred images")
    #my_applier.plot_batches_of_5()

    # Plot batches of 10 of trajectory,
    # PSF and blurred image
    #   my_applier.plot_batches_of_10()
    # my_applier.save_plot_to_image('C:\\Users\\Pavel\\Desktop\\DIP_hw1_repo\\results\\psfs\\')

    # Get a list of blurred images
    blurred_images = my_applier.get_blurred_images()

    # Fix the blurred images
    my_fixer = Blurr_Fixer(blurred_images,p=10,ifft_scale=995,
                               original_size=256, margin=6)
    fixed = my_fixer.fix_blurr()
    #my_fixer.show_Fixed()

    print("PLot n to PSNR graph")

    num_samples = list()
    PSNR_results = list()
    fixed_images = list()

    # Iterate over different number of blurred images,
    # Calc the PSNR and show the result
    for n in range(1,101,1):
        print("Deblurring for ",n," samples...")
        num_samples.append(n)
        my_applier = Blurr_Applier(n)
        my_applier.generate_PSFs()
        my_applier.apply_blurr()

        blurred_images = my_applier.get_blurred_images()

        my_fixer = Blurr_Fixer(blurred_images, p=10,ifft_scale=995,
                               original_size=256, margin=6)
        fixed = my_fixer.fix_blurr()
        fixed_images.append(fixed)
        my_calc = PSNR_calculator(my_Im.get_image(), fixed)
        PSNR_results.append(my_calc.evaluate_PSNR())

    plt.plot(num_samples, PSNR_results)
    plt.xlabel("Number of blurred samples")
    plt.ylabel("PSNR [dB]")
    plt.savefig("C:\\Users\\Pavel\\Desktop\\DIP_hw1_repo\\results\\psnr_graph\\psnr_graph.png")
    plt.show()

    # save to file PSNR + images
    for i in range(len(fixed_images)):
        cropped = fixed_images[i]
        plt.imshow(cropped, cmap='gray')
        k = i+1
        plt.title("n={0}, PSNR={1}".format(k,PSNR_results[i]), fontsize=10)
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        path = "C:\\Users\\Pavel\\Desktop\\DIP_hw1_repo\\results\\psnr_images\\"
        full_path = path + str(i) + '.png'
        plt.savefig(full_path)


    for i in range(len(fixed_images)):
        plt.subplot(10, 10, i+1)
        cropped = fixed_images[i][25:100, 100:175]
        plt.imshow(cropped, cmap='gray')
        k = i+1
        plt.ylabel("n=%i" % k, fontsize=5)
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
    plt.show()

    # Show first and last iteration for comparison.
    #plt.subplot(1, 3, 1)
    #cropped = blurred_images[0]
    #plt.imshow(cropped, cmap='gray')
    #plt.title("First Blurred image")
    #plt.subplot(1, 3, 2)
    #cropped = fixed_images[0]
    #plt.imshow(cropped, cmap='gray')
    #plt.title("First iteration")
    #plt.subplot(1, 3, 3)
    #cropped = fixed
    #plt.imshow(cropped, cmap='gray')
    #plt.title("100th iteration")
    #plt.show()


if __name__ == '__main__':
    main()