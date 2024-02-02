import cv2 as cv
import numpy as np

sift = cv.xfeatures2d_SIFT.create()

vocabulary = np.load('vocabulary.npy')

# Load SVM
svm1 = cv.ml.SVM_create() #load apo to svm meta to create
svm1 = svm1.load('svm1')

svm2 = cv.ml.SVM_create() #load apo to svm meta to create
svm2 = svm2.load('svm2')

svm3 = cv.ml.SVM_create() #load apo to svm meta to create
svm3 = svm3.load('svm3')

svm4 = cv.ml.SVM_create() #load apo to svm meta to create
svm4 = svm4.load('svm4')

svm5 = cv.ml.SVM_create() #load apo to svm meta to create
svm5 = svm5.load('svm5')
# Classification
descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

testpaths = np.load('testpaths.npy')
responsefinal = []
filesinclass0 = 0
filesinclass1 = 0
filesinclass2 = 0
filesinclass3 = 0
filesinclass4 = 0
correctmoto=0
correctbus=0
correctbike=0
correctairplane=0
correctcarside=0
for p in testpaths: #this is for testing every image of the test folders
    print('')
    print('---image in path:',p,'---')
    img = cv.imread(p)
    kp = sift.detect(img)
    bow_desc = descriptor_extractor.compute(img, kp)#bag of words descriptor 50 theseon
    testpaths = np.load('testpaths.npy')
    response1 = svm1.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    response2 = svm2.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    response3 = svm3.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    response4 = svm4.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    response5 = svm5.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    response = [response1[1], response2[1], response3[1], response4[1], response5[1]]
    responsefinal.append(response)
    #print(response)
    for j in range(0, 4):
        for i in range(0, 4):
            if (response[i] < 0 and response[j] < 0):
                a = abs(response[i])
                b = abs(response[j])
                if a > b:
                    result = i
                else:
                    result = j
    if response[i] < 0:
            if i==0 :
                print("It's a motorbike!")
            elif i==1 :
                print("It's a school bus!")
            elif i==2 :
                print("It's a touring bike!")
            elif i==3 :
                print("It's an airplane!")
            elif i==4 :
                print("It's a car side!")

    else:
            result = np.argmin(response)
            print("All responses positive,the response with the min distance from hyperlevel is:", result)
            if result == 0:
                print("It's a motorbike!")
            elif result == 1:
                print("It's a school bus!")
            elif result == 2:
                print("It's a touring bike!")
            elif result == 3:
                print("It's an airplane!")
            elif result == 4:
                print("It's a car side!")


    #this is ONLY for checking acurracy, no use for the algorithm!
    if 'caltech/imagedb_test/145.motorbikes-101' in p:
        filesinclass0=filesinclass0+1
        if response[0]<0 or result==0:
            correctmoto=correctmoto+1

    elif 'caltech/imagedb_test/178.school-bus' in p:
        filesinclass1= filesinclass1 + 1
        if response[1]<0 or result==1:
            correctbus=correctbus+1

    elif 'caltech/imagedb_test/224.touring-bike' in p:
        filesinclass2 = filesinclass2 + 1
        if response[2]<0 or result==3:
            correctbike=correctbike+1

    elif 'caltech/imagedb_test/251.airplanes-101' in p:
        filesinclass3= filesinclass3 + 1
        if response[3]<0 or result==3:
            correctairplane=correctairplane+1

    else:
        filesinclass4= filesinclass4 + 1
        if response[4]<0 or result==4:
            correctcarside=correctcarside+1

totalcorrect=correctairplane+correctmoto+correctcarside+correctbike+correctbus
print(totalcorrect)


def accuracy(a,b,foundcorrect):
 foundwrong = b - a - foundcorrect
 print('Total files:',b-a)
 print('Number of images with wrong prediction:',foundwrong)
 print('Number of images with correct prediction:',foundcorrect)
 acc=(foundcorrect/(b-a))*100
 print('The accuracy is:',acc)
 return acc


print('')
print('---Checking total accuracy of algorithm---')
acc=accuracy(0,testpaths.shape[0],totalcorrect)
print('')
print('---Checking accuracy for motorbike---')
accofclass0=accuracy(0,filesinclass0,correctmoto)
print('')
print('---Checking accuracy for schoolbus---')
accofclass1=accuracy(filesinclass0,filesinclass0+filesinclass1,correctbus)
print('')
print('---Checking accuracy for touringbike---')
accofclass2=accuracy(filesinclass0+filesinclass1,filesinclass0+filesinclass1+filesinclass2,correctbike)
print('')
print('---Checking accuracy for airplanes---')
accofclass3=accuracy(filesinclass0+filesinclass1+filesinclass2,filesinclass0+filesinclass1+filesinclass2+filesinclass3,correctairplane)
print('')
print('---Checking accuracy for carside---')
accofclass4=accuracy(filesinclass0+filesinclass1+filesinclass2+filesinclass3,filesinclass0+filesinclass1+filesinclass2+filesinclass3+filesinclass4,correctcarside)



pass