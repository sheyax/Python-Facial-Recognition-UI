#Import required libraries 
from tkinter import *
import cv2
from tkinter import messagebox
import os
import numpy 
from pathlib import Path
#define a class for facial detection

class FaceDetect():
    #initialize the class with
    #go to bottom of the code to define root 
    def __init__(self, root):
        self.root = root
        self.haar_file= 'haarcascade_frontalface_default.xml'
        
        #Main Windows construct
        Label(self.root, bg='white', height=2).pack(fill=BOTH)
        title= Label(self.root, bg='#0099ff', 
                     text=' Face Detect', 
                     font=('arial', 15, 'bold'), 
                     height=3, bd=2, relief='groove')
        title.pack(fill=BOTH)      
        
        #control window boundary
        control_frame= Frame(self.root, height=200, bg='white', bd=4, relief='ridge')  
        control_frame.pack(pady=20, fill=BOTH, padx=10)
        
        #Build the buttons 
        train_button= Button(control_frame, text= 'Train Model',
                             bd=2, height= 3,width= 10, relief=GROOVE, 
                             font=('arial', 12, 'bold'), 
                             command=self.get_data)
        #position of the button
        train_button.place(x=60, y=50)
        
        #test model button
        test_button= Button(control_frame, text= 'Test Model',
                             bd=2, height= 3, width= 10, relief=GROOVE, 
                             font=('arial', 12, 'bold'), 
                             command=self.test_model)
        #position of the button
        test_button.place(x=220, y=50)
        
        #exit button
        exit_button= Button(control_frame, text= 'Exit',
                             bd=2, height= 3, width= 10, relief=GROOVE, 
                             font=('arial', 12, 'bold'), 
                             command=root.quit)
        #position of the button
        exit_button.place(x=370, y=50)
        
    
    def train_model(self):
        #fetch name and id----------------------------------------------------------------
        name_= self.name.get()
        id_= self.id_ent.get()
        print(name_, id_)
        self.top.destroy()
        self.take_images(name_, id_)
        
        
        
    def get_data(self):
        self.top= Toplevel()
        self.top.geometry('300x200+240+200')
        self.top.configure(bg='#0099ff')
        self.top.resizable(0,0)
        
        #name input 
        name_lbl= Label(self.top, text='Your Name', width=10, font=('arial', 10, 'bold')).place(x=10, y=20)
        self.name= Entry(self.top, width= 15, font=('arial', 12))
        self.name.place(x=120, y=20)
        
        #id input
        id_lbl= Label(self.top, text='Your ID', width=10, 
                      font=('arial', 10, 'bold')).place(x=10, y=60)
        self.id_ent= Entry(self.top, width= 15, font=('arial', 12))
        self.id_ent.place(x=120, y=60)
        
        #btn input 
        btn= Button(self.top, text='Train Model', font=('arial', 12, 'bold'), 
                    command=self.train_model)
        btn.place(x=100, y=120)
        
        #Train the model based on save images and  
    def test_model(self):
        datasets='dataset'
        #create a list of images and a list of corresponding names 
        (images, labels, names, id)= ([],[],{}, 0)
        for(subdirs, dirs, files)in os.walk(datasets):
            for subdir in dirs:
                names[id]= subdir
                subjectpath= os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path= subjectpath+ '/'+ filename
                    label=id
                    images.append(cv2.imread(path, 0))
                    labels.append(int(label))
                id += 10
        width, height= (130, 100)
        
        #Create a numpy array from two lists above 
        (images, labels) =[numpy.array(lis) for lis in [images, labels]]
        
        #OpenCV train models from images 
        model= cv2.face.LBPHFaceRecognizer_create()
        model.train(images, labels)
        
        #Stream Camera view button
        face_cascade= cv2.CascadeClassifier(self.haar_file)
        webcam= cv2.VideoCapture(0)
        while True:
            (_,im)= webcam.read()
            gray= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces= face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                
            #Try to recognize face
                prediction= model.predict(face_resize)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            
                if prediction[1] < 500:
                    cv2.putText(im, '% s - %.0f' %
                                (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                else:
                    cv2.putText(im, 'not recognized',
                                (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            cv2.imshow('OpenCV', im)

            key = cv2.waitKey(10)
            if key == 27:
                break
        cv2.destroyAllWindows()
    
    def take_images(self,name_,id_):
        # time.sleep(2)
        # All the faces data will be
        # present this folder
        datasets = 'dataset'
        # These are sub data sets of folder,
        # for my faces I've used my name you can
        # change the label here
        sub_data = str(name_)+ '-' + str(id_)
        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)

        # defining the size of images
        (width, height) = (130, 100)

        # '0' is used for my webcam,
        # if you've any other camera
        # attached use '1' like this
        face_cascade = cv2.CascadeClassifier(self.haar_file)
        webcam = cv2.VideoCapture(0)

        # The program loops until it has 30 images of the face.
        count = 1
        while count < 120:
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('% s/% s.png' % (path, count), face_resize)
            count += 1

            cv2.imshow('OpenCV', im)
            key = cv2.waitKey(20)
            if key == 37:
                break
        cv2.destroyAllWindows()
        messagebox.showinfo("Python Says","Model is Trained with Your \n  Image Data")



#define the main windows GUI
if __name__== '__main__':
    root= Tk()
    FaceDetect(root)
    root.geometry('550x330+240+200')
    root.title('Real time face detector review')
    root.resizable(0, 0)
    root.configure(bg='#0099ff')
    root.mainloop()
    