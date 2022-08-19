from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity



feature_list=np.array(pickle.load(open('feature.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))

model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

detector=MTCNN()

sample_img= cv.imread('Uploads/Ajaydevgan-lal.jpg')
det_face= detector.detect_faces(sample_img)

x,y,width,height=det_face[0]['box']

face=sample_img[y:y+height,x:x+width]

img=Image.fromarray(face)
img=img.resize((224,224))

face_img = np.array(img)
face_img = face_img.astype('float32')

expanded_image = np.expand_dims(face_img,axis=0)
pre_img=preprocess_input(expanded_image)

result=model.predict(pre_img).flatten()

similarity=[]

for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

index_pos= sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

temp_img=cv.imread(filenames[index_pos])
cv.imshow('predicted',temp_img)
cv.waitKey(0)