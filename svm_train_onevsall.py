import cv2 as cv
import numpy as np

train_folders = ['145.motorbikes-101', '178.school-bus','224.touring-bike','251.airplanes-101','252.car-side-101']


sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

bow_descs = np.load('index.npy').astype(np.float32)

img_paths = np.load('paths.npy')

# Train SVM
print('Training SVM...')
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)#rbf kernel for non liner problems
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))


labels1= []
labels2= []
labels3= []
labels4= []
labels5= []

for p in img_paths:
    if '145.motorbikes-101' in p:
        labels1.append(1)
    else:
        labels1.append(0)
labels1 = np.array(labels1, np.int32)

for p in img_paths:
    if '178.school-bus' in p:
        labels2.append(1)
    else:
        labels2.append(0)
labels2 = np.array(labels2, np.int32)

for p in img_paths:
    if '224.touring-bike' in p:
        labels3.append(1)
    else:
        labels3.append(0)
labels3 = np.array(labels3, np.int32)

for p in img_paths:
    if '251.airplanes-101' in p:
        labels4.append(1)
    else:
        labels4.append(0)
labels4 = np.array(labels4, np.int32)

for p in img_paths:
    if '252.car-side-101' in p:
        labels5.append(1)
    else:
        labels5.append(0)
labels5 = np.array(labels5, np.int32)


svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels1)#kathe sample pou tha mou doseis einai mia grammi!(ROW_SAMPLE)
svm.save('svm1')
svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels2)#kathe sample pou tha mou doseis einai mia grammi!(ROW_SAMPLE)
svm.save('svm2')
svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels3)#kathe sample pou tha mou doseis einai mia grammi!(ROW_SAMPLE)
svm.save('svm3')
svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels4)#kathe sample pou tha mou doseis einai mia grammi!(ROW_SAMPLE)
svm.save('svm4')
svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels5)#kathe sample pou tha mou doseis einai mia grammi!(ROW_SAMPLE)
svm.save('svm5')

