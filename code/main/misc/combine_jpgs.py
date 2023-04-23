from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    master_dir = "E:\\PhD\\results\\time_domain\\56_256ms\\habituation\\partitions\\headache\\"
    highlighted_electrode = master_dir +  "highlighted_electrode.png"
    cluster_through_time = master_dir + "positive_cluster.png"
    erps = master_dir + "positive_peak_erp_1.png"
    topographic_maps = master_dir + "positive_topographic.png"

    # read two input images
    img1 = cv2.imread(cluster_through_time)
    img2 = cv2.imread(topographic_maps)
    img3 = cv2.imread(erps)


    # both image height and width should be same
    img1 = cv2.resize(img1, (5000, 3000))
    img2 = cv2.resize(img2, (5000, 3000))
    img3 = cv2.resize(img3, (5000, 4000))
    padding = np.zeros((100, 5000, 3), np.uint8)
    padding[:] = 255

    # join the two images vertically
    img = cv2.vconcat([img1, padding, img2, padding, img3])

    # Convert the BGR image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.savefig('test.png', dpi=100)
    plt.close()