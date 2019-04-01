import spacy
import pytesseract
from pytesseract import Output
import cv2
from langdetect import detect_langs
from langdetect import DetectorFactory 


#1.lecturer detection (if needed)

#from imageai.Detection import ObjectDetection
#import os
#execution_path = os.getcwd()
#
#detector = ObjectDetection()
#detector.setModelTypeAsRetinaNet()
#detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
#detector.loadModel()
#detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "1.jpg"), 
#             output_image_path=os.path.join(execution_path , "imagenew.jpg"), 
#             minimum_percentage_probability=72)
#
#for eachObject in detections:
#    if eachObject["percentage_probability"] > 70:
#        print(eachObject["name"] , " : " , eachObject["percentage_probability"])
#    

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better


#2.colour clustering in 2 colours (if needed)

#img = cv2.imread('2.jpg')
#h1, w1, _ = img.shape
#image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#orig = image.copy()
#reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
#kmeans = KMeans(n_clusters = 2)
#kmeans.fit(reshaped)
#
#clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
#    (image.shape[0], image.shape[1]))
#
#sortedLabels = sorted([n for n in range(2)],
#    key=lambda x: -np.sum(clustering == x))
#
#kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
#for i, label in enumerate(sortedLabels):
#    kmeansImage[clustering == label] = int((255) / (2 - 1)) * i
#
#
#
#
#
#cv2.imshow('Original vs clustered', kmeansImage)
#cv2.waitKey(0)

#3.OCR for obtaining bounding boxes

img = cv2.imread('3 (1).jpg')
height, width, channels = img.shape
DetectorFactory.seed = 0
d = pytesseract.image_to_data(img, output_type=Output.DICT)
numberBoxes = len(d['level'])

#Merge bounding boxes from the same block
#This solves the overlapping
rectangles = [[0 for i in range(4)] for y in range(1000)] 

for i in range(numberBoxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    
    if x == 0 and y == 0:
        rectangles[d['block_num'][i]] = (0, 0, 0, 0)
        continue
            
    if w * h > rectangles[d['block_num'][i]][2] * rectangles[d['block_num'][i]][3]:
            rectangles[d['block_num'][i]] = (x, y, w, h)

#Draw the rectangles on the image
for i in range(numberBoxes):
    (x, y, w, h) = rectangles[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#Show the image on screen
cv2.imshow('img', img)
cv2.waitKey(0)

#Merge the text in only one string
noLastElement = 0
mergedText = ""
listMergedText = []
for i in range(numberBoxes):
    noLastElement = 0
    if d['block_num'][i] != d['block_num'][i - 1] and i > 0:
        if mergedText.startswith(' '):
            mergedText = mergedText[1:]
        listMergedText.append(mergedText)
        mergedText = ""
        noLastElement = 1
    else:
        if mergedText.endswith(' '):
            mergedText += d['text'][i]
        else:
            mergedText += ' ' + d['text'][i]
            
if noLastElement == 0:
    listMergedText.append(mergedText)

#Use SpaCy to compute perplexity for each word
#A better trained nlp for these domain would improve the results

nlp = spacy.load('es_core_news_md')
id = 0
textArea = 1
imageArea = 1
nrTextBoxes = 0
nrImageBoxes = 0

# latin languages + english
latin = ["es", "ca", "fr", "it", "lt", "pt", "ro", "en"]

#Use SpaCy + langdetect to see if a box is an image or a text
for s in listMergedText:
    isText = 0
    area = rectangles[id][2] * rectangles[id][3]
    id += 1
    doc = nlp(s)

    for token in doc:
       try:
           checker = 0
           s1 = detect_langs(token.text)
           for languages in s1:
               if str(languages)[:2] in latin:
                      checker += 1
                      break
                      
           if token.prob > -16: #threshold for word perplexity(range: [-20; 0])
               checker += 1
           if checker == 2:
               isText += 1
           
       except:
          if token.prob < -3: #threshold for punctuation perplexity
              isText -= 1
    if isText <= 0:
        imageArea += area
        nrImageBoxes += 1
    else:
        textArea += area
        nrTextBoxes += 1
       

#The percetages are computed from the whole area detected, 
#not the whole picture
print ("image: ", imageArea/(imageArea + textArea) * 100, 
       "text: ", 100 - imageArea/(imageArea + textArea) * 100)
print ("image: ", nrImageBoxes, imageArea, " text: ", nrTextBoxes, textArea)

