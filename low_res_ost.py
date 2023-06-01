from imresize import imresize
import imageio
import os
import glob

def imread(img_path):
    img = imageio.imread(img_path)
    if len(img.shape) == 2:
        img = np.stack([img, ] * 3, axis=2)
    return img

paths = sorted(glob.glob(os.path.join('/storage/sr_data/4/hr/ost_test', '*')))

for idx, path in enumerate(paths):
    imgname, extension = os.path.splitext(os.path.basename(path))

    img = imread(path)
    resized_img = imresize(img, scalar_scale=0.25)
    save_path = '/storage/sr_data/4/lr/ost_test/' + os.path.basename(path)
    print(save_path)
    # imageio.imwrite(save_path, resized_img)
