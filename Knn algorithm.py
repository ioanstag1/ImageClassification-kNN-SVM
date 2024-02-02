import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


bow_descs = np.load('index.npy').astype(np.float32)

img_paths = np.load('paths.npy')

# TRAINING
labels = []


train_folders = ['caltech/imagedb/145.motorbikes-101', 'caltech/imagedb/178.school-bus','caltech/imagedb/224.touring-bike','caltech/imagedb/251.airplanes-101','252.car-side-101']
for p in img_paths:
    if 'caltech/imagedb/145.motorbikes-101' in p:
        labels.append(0)
    elif 'caltech/imagedb/178.school-bus' in p:
        labels.append(1)
    elif 'caltech/imagedb/224.touring-bike' in p:
        labels.append(2)
    elif 'caltech/imagedb/251.airplanes-101' in p:
        labels.append(3)
    else:
        labels.append(4)
labels = np.array(labels, np.int32)


def knnalg(desc,n,bow_descs):
    classes=np.zeros(5)
    mindist=np.zeros(n)
    for k in range(desc.shape[0]):
        distances1 = np.sum((desc[k, :] - bow_descs) ** 2, axis=1)
        mini = np.argsort(distances1)#position of sorted distances
        mindist=mini[0:n]
        #append the classes of the closest descriptors
        for i in mindist:
            if labels[i]==0: #for class 0
                classes[0]+=1
            elif labels[i]==1: #for class 1
                classes[1]+=1
            elif labels[i]==2: #for class 2
                classes[2]+=1
            elif labels[i]==3: #for class 3
                classes[3]+=1
            elif labels[i]==4: #for class 4
                classes[4]+=1
    results=np.argmax(classes)#find the position(class) that has the max value
    #max value is the class that contains the more descriptors that are closest to the desc of test img
    #print(classes)
    return results



#Function for priting images with their label
def display_image_with_label(img,label):
    text_position = (10, 30)  # Coordinates of the text position
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 0, 255)  # Red color in BGR format
    thickness = 2
    cv.putText(img, label, text_position, font, font_scale, font_color, thickness)

    # Display the image
    cv.imshow("Image with Label", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# TESTING
sift = cv.xfeatures2d_SIFT.create()

vocabulary = np.load('vocabulary.npy')

descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

#i kainourgia eikona apo to test den einai epeksergasmeni proigoumenos

testpaths = np.load('testpaths.npy')
correctresults=[]
resultsfinal = []
filesinclass0=0
filesinclass1=0
filesinclass2=0
filesinclass3=0
filesinclass4=0

for p in testpaths: #this is for testing every image of the test folders
    img = cv.imread(p)
    kp = sift.detect(img) #keypoint detection for every test image
    bow_desc = descriptor_extractor.compute(img, kp)# BOW descriptor for every test image

    results = knnalg(bow_desc, 12, bow_descs)
    resultsfinal.append(results)
    if results == 0:

        label="It's a motorbike!"
        #display_image_with_label(img,label)
    elif results == 1:
        label="It's a school bus!"
        #display_image_with_label(img, label)
    elif results == 2:
        label="It's a touring bike!"
        #display_image_with_label(img, label)
    elif results == 3:
        label="It's an airplane!"
        #display_image_with_label(img, label)
    else:
        label="It's a car side!"
        #display_image_with_label(img, label)
    #algorithm stops here
    #this is ONLY for checking acurracy, no use for the algorithm!
    if 'caltech/imagedb_test/145.motorbikes-101' in p:
        correctresults.append(0)
        filesinclass0=filesinclass0+1
    elif 'caltech/imagedb_test/178.school-bus' in p:
        correctresults.append(1)
        filesinclass1= filesinclass1 + 1

    elif 'caltech/imagedb_test/224.touring-bike' in p:
        correctresults.append(2)
        filesinclass2 = filesinclass2 + 1

    elif 'caltech/imagedb_test/251.airplanes-101' in p:
        correctresults.append(3)
        filesinclass3= filesinclass3 + 1

    else:
        correctresults.append(4)
        filesinclass4= filesinclass4 + 1

resultsfinal = np.array(resultsfinal, np.int32)
correctresults = np.array(correctresults, np.int32)

# print(filesinclass0)
# print(filesinclass1)
# print(filesinclass2)
# print(filesinclass3)
# print(filesinclass4)

def accuracy(a,b):
 foundcorrect = 0
 foundwrong = 0


 for i in range(a,b):
    if resultsfinal[i]==correctresults[i]:
            foundcorrect=foundcorrect+1
    else:
            foundwrong=foundwrong+1


 print('Total files:',b-a)
 print('Number of images with wrong prediction:',foundwrong)
 print('Number of images with correct prediction:',foundcorrect)
 acc=(foundcorrect/(b-a))*100
 print('The accuracy is:',acc)
 return acc


def histogram(a,b,label):
    # Define the labels of your classes
    labels = ['motorbike', 'school bus', 'touring bike', 'airplane', 'car side']

    # Count the number of times each class appears in the predictions
    counts = np.zeros(len(labels))
    for i in range (a,b):
        counts[resultsfinal[i]] += 1
    # Create the histogram

    plt.bar(labels, counts)

    # Set the title and labels
    plt.title(label)
    plt.xlabel('Classes')
    plt.ylabel('Counts')

    # Show the plot
    plt.show()




# Define the labels of your classes
labels = ['Motorbike', 'School Bus', 'Touring Bike', 'Airplane', 'Car Side']
# Define the start and end indices for each class
start_indices = [0, filesinclass0, filesinclass0 + filesinclass1, filesinclass0 + filesinclass1 + filesinclass2, filesinclass0 + filesinclass1 + filesinclass2 + filesinclass3]
end_indices = [filesinclass0, filesinclass0 + filesinclass1, filesinclass0 + filesinclass1 + filesinclass2, filesinclass0 + filesinclass1 + filesinclass2 + filesinclass3, filesinclass0 + filesinclass1 + filesinclass2 + filesinclass3 + filesinclass4]

print('')
print('---Checking total accuracy of algorithm---')
acc = accuracy(0, len(resultsfinal))
print('Total Accuracy:', acc)

# Loop through each class
for i in range(len(start_indices)):
    class_label = labels[i]
    start_index = start_indices[i]
    end_index = end_indices[i]

    print('')
    print(f'---Checking accuracy for {class_label}---')
    acc_of_class = accuracy(start_index, end_index)
    #histogram(start_index, end_index, f'Predicted Classes Histogram for {class_label}')




