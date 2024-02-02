import os
import cv2 as cv
import numpy as np

img_paths = []
test_folders = ['caltech/imagedb_test/145.motorbikes-101', 'caltech/imagedb_test/178.school-bus','caltech/imagedb_test/224.touring-bike','caltech/imagedb_test/251.airplanes-101','caltech/imagedb_test/252.car-side-101']
for folder in test_folders:
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        img_paths.append(path)

np.save('testpaths', img_paths)
