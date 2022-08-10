import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from numpy import asarray
from skimage import io
# import Image


# image = data.astronaut()
# print("image: ", image)
# print("image_TYPE: ", image.shape)
# image = "C:\\Users\\HP\\Desktop\\fiverr\\Task_covid_Detection\\Spyder code files-20220806T171740Z-001\\Spyder code files\\archive (1)\\COVID-19_Radiography_Dataset\\COVID\images\\COVID-4.png"
# image = ".\\archive (1)\\COVID-19_Radiography_Dataset\\COVID\\images\\COVID-13.png"
def get_hog(image):
    r = image.split("\\")[-1]
    print(r)


    image = io.imread(image)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=None)



    fig, ax2 = plt.subplots(1,1, figsize=(12,12), sharex=True, sharey=True)

    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    # print(type(ax2))

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    fig.savefig('.\static\images\hog\hog_'+r)
    # plt.show()

# get_hog(image)