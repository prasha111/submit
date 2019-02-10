import cv2
import numpy as np

# Load HAAR face classifier
b=0
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = '/home/prashant/Downloads/t/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")




from os import listdir
from os.path import isfile, join
print(cv2.__version__)
# Get the training data we previously made
data_path = '/home/prashant/Downloads/t/'
# a=listdir('d:/faces')
# print(a)
# """
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
# 
# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)
model=cv2.face_LBPHFaceRecognizer.create()
# Initialize facial recognizer
# model = cv2.face_LBPHFaceRecognizer.create()
# model=cv2.f
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()

# Let's train our model 
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")

#import cv2
#import numpy as np
import webbrowser


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
        print(results)
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 75:
            cv2.putText(image, "Hey prashant", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )

            b+=12

            if b>100:

                import pyttsx3
                import speech_recognition  as sr
                import webbrowser as wb
                speaker = pyttsx3.init()
                speaker.say("hi you    haaave loogin ")
                speaker.runAndWait()

                speaker.say("what you wanna do further  ")

                speaker.runAndWait()

                speaker.say("enter ` for cheking summary from web\n enter 2 for taking summary of a jpg pic \n ")

                p = int(input("enter 1 for cheking summary from web\n enter 2 for taking summary of a jpg pic  \n"))

                if p==1:



                    import bs4 as bs  
                    #import urllib.request  
                    import re
                    import urllib
                    import nltk

                    nltk.download('punkt')
                    nltk.download('stopwords')

                    inputk = raw_input("enter the input url you want to search or you can paste it here\n")
                    scraped_data = urllib.urlopen(inputk) 

                    #scraped_data = urllib.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')

                    article = scraped_data.read()

                    parsed_article = bs.BeautifulSoup(article,'lxml')

                    paragraphs = parsed_article.find_all('p')

                    article_text = ""

                    for p in paragraphs:  
                        article_text += p.text

                    # Removing Square Brackets and Extra Spaces
                    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
                    article_text = re.sub(r'\s+', ' ', article_text)  

                    # Removing special characters and digitsaq
                    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                    sentence_list = nltk.sent_tokenize(article_text)

                    stopwords = nltk.corpus.stopwords.words('english')

                    word_frequencies = {}  
                    for word in nltk.word_tokenize(formatted_article_text):  
                        if word not in stopwords:
                            if word not in word_frequencies.keys():
                                word_frequencies[word] = 1
                            else:
                                word_frequencies[word] += 1

                    maximum_frequncy = max(word_frequencies.values())
                    for word in word_frequencies.keys():  
                        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                    sentence_scores = {}  
                    for sent in sentence_list:  
                        for word in nltk.word_tokenize(sent.lower()):
                            if word in word_frequencies.keys():
                                if len(sent.split(' ')) < 30:
                                    if sent not in sentence_scores.keys():
                                        sentence_scores[sent] = word_frequencies[word]
                                    else:
                                        sentence_scores[sent] += word_frequencies[word]
                    import heapq  
                    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
                    summary = ' '.join(summary_sentences)  
                    print(summary)
                    speaker.say("if you want to say it orally the text press 8")
                    speaker.runAndWait()
                    kk= int(input("if you want to say it orally the text press 8 \n"))
                    if kk == 8 :
                        speaker.say(summary)
                        speaker.runAndWait()


                    import pyttsx3
                    import speech_recognition  as sr
                    import webbrowser as wb
                    speaker = pyttsx3.init()
                    #speaker.say("hi what you wanana do")
                    #speaker.runAndWait()
                    #mic  =  sr.Microphone()
                    #r = sr.Recognizer()
                    speaker.say(summary)
                    speaker.runAndWait()



                if p==2:

                    url = "http://192.168.43.1:8080/shot.jpg"
                    import requests
                    import cv2
                    import numpy
                    import Image
                    import pytesseract
                    while True:
                        img_location = requests.get(url)
                        img_content =  img_location.content
                        img_content_array = bytearray(img_content)
                        img_1d = numpy.array(img_content_array)
                        img = cv2.imdecode(img_1d, -1 )
                        cv2.imshow('hi', img)
                        rrr=(pytesseract.image_to_string(img))



                        """scraped_data = urllib.urlopen(rrr) 

                        #scraped_data = urllib.urlopen('')"""

                        article = rrr.read()

                        parsed_article = bs.BeautifulSoup(article,'lxml')

                        paragraphs = parsed_article.find_all('p')

                        article_text = ""

                        for p in paragraphs:  
                            article_text += p.text

                        # Removing Square Brackets and Extra Spaces
                        article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
                        article_text = re.sub(r'\s+', ' ', article_text)  

                        # Removing special characters and digitsaq
                        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                        sentence_list = nltk.sent_tokenize(article_text)

                        stopwords = nltk.corpus.stopwords.words('english')

                        word_frequencies = {}  
                        for word in nltk.word_tokenize(formatted_article_text):  
                            if word not in stopwords:
                                if word not in word_frequencies.keys():
                                    word_frequencies[word] = 1
                                else:
                                    word_frequencies[word] += 1

                        maximum_frequncy = max(word_frequencies.values())
                        for word in word_frequencies.keys():  
                            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                        sentence_scores = {}  
                        for sent in sentence_list:  
                            for word in nltk.word_tokenize(sent.lower()):
                                if word in word_frequencies.keys():
                                    if len(sent.split(' ')) < 30:
                                        if sent not in sentence_scores.keys():
                                            sentence_scores[sent] = word_frequencies[word]
                                        else:
                                            sentence_scores[sent] += word_frequencies[word]
                        import heapq  
                        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
                        summary = ' '.join(summary_sentences)  
                        print(summary)
                        speaker.say("if you want to say it orally the text press 8")
                        speaker.runAndWait()
                        kk= int(input("if you want to say it orally the text press 8 \n"))
                        if kk == 8 :
                            speaker.say(summary)
                            speaker.runAndWait()


                        import pyttsx3
                        import speech_recognition  as sr
                        import webbrowser as wb
                        speaker = pyttsx3.init()
                        #speaker.say("hi what you wanana do")
                        #speaker.runAndWait()
                        #mic  =  sr.Microphone()
                        #r = sr.Recognizer()
                        speaker.say(summary)
                        speaker.runAndWait()















                       
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()   
