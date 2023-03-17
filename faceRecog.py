from PIL import Image as Img
from numpy import asarray
from numpy import expand_dims
import numpy as np
import pandas as pd

import cv2
from keras_facenet import FaceNet

from os import listdir

folder='fotos/'

def cropIne(photo):
  "Recorto la mitad derecha de la ine para evitar el segundo rostro"
  rows, cols, _ = photo.shape
  rows1percent = rows/100
  cols1percent = cols/100
  leftHalf = photo[round(rows1percent*0): round(rows1percent*100), round(cols1percent*0): round(cols1percent*50)].copy()
  return leftHalf

#TODO: findSimilarFace sólo busca en database, y mide distancias
#necesito actualizar database cada que se agrega un voluntario, para eso existe add_face()
def findSimilarFace(filename, threshold = 0.7):
  DB = pd.read_csv('database.csv')
  photo = cv2.imread(folder + filename)
  photo = cropIne(photo)
  signature = photoToVector(photo)  
  min_dist=100
  identity=' '
  DB = DB.T
  
  for index, row in DB.items() :
    key = row[0]
    value = row[1:513]
    value = value.to_numpy()
    if key!=filename:
        dist = np.linalg.norm(value-signature)
        #print(dist)
        if dist < min_dist:
            min_dist = dist
            identity = key
  
  if min_dist < threshold:
      return identity
  else:
      return None
  
def photoToVector(photo):
    HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
    MyFaceNet = FaceNet()
    
    gbr = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    gbr = Img.fromarray(gbr)                  # konversi dari OpenCV ke PIL
    gbr_array = asarray(gbr)
   
    wajah = HaarCascade.detectMultiScale(photo,1.1,4)
    for (x1,y1,w,h) in wajah:
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + w, y1 + h
      
        face = gbr_array[y1:y2, x1:x2]                        
      
        face = Img.fromarray(face)                       
        face = face.resize((160,160))
        face = asarray(face)
      
        face = expand_dims(face, axis=0)
        signature = MyFaceNet.embeddings(face)
    return signature

#TODO: CSV es lentisimo, utilizar pickle o parquet
def create_database():
    # Create header of the csv file
    vector_names = ['v' + str(x) for x in range(1, 513)]
    columns = ['name'] + vector_names

    df = pd.DataFrame(columns=columns)
    df.to_csv('database.csv', index=False)

def add_face(filename):
    database = pd.read_csv('database.csv')
    img = cv2.imread(folder + filename)

    img = cropIne(img)
    emb = photoToVector(img)[0]
    register = [filename] + emb.tolist()
    database = database.append(pd.Series(register, index=database.columns), ignore_index=True)

    # Save to csv file
    database.to_csv('database.csv', index=False)

def add_all_faces():
    for filename in listdir(folder):
        add_face(filename)

def set_folder(fldr):
    global folder
    folder = fldr.replace(".", "/")
    
def get_folder():
    return folder
#create_database()
#add_all_faces()
#add_face("Zani.jpeg")

#------------------------------------API---------------------------------------

def findSimilar(filename):
    try:
        message = findSimilarFace(filename)
        return message
    except Exception as err:
      print(str(err))
      return str(err)
      
def resetDatabase():
    try:
        create_database()
        return 200
    except Exception as err:
      print(str(err))
      return str(err)
      
def add(filename):
    try:
        add_face(filename)
        return 200
    except Exception as err:
      print(str(err))
      return str(err)

def addAll():
    try:
        add_all_faces()
        return 200
    except Exception as err:
      print(str(err))
      return str(err)
  
def setFolder(fldr):
    try:
        set_folder(fldr)
        return 200
    except Exception as err:
      print(str(err))
      return str(err)

def getFolder():
    try:
        return get_folder()
    except Exception as err:
      print(str(err))
      return str(err)