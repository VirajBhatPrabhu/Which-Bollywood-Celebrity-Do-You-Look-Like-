from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os

feature_list=pickle.load(open('feature.pkl','rb'))
filenames=pickle.load(open('filenames.pkl','rb'))

model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

detector=MTCNN()

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('Uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False



def extract_features(img_path,model,detector):
    img = cv.imread(img_path)
    results = detector.detect_faces(img)

    x,y,width,height = results[0]['box']

    face = img[y:y+height, x:x+width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_img = np.asarray(image)
    face_img = face_img.astype('float32')

    expanded_image = np.expand_dims(face_img, axis=0)
    pre_img = preprocess_input(expanded_image)

    result = model.predict(pre_img).flatten()
    return result


def recommend(feature_list,features):
    similarity = []

    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


st.title('Which Celebrity Do You Look Like ?')

uploaded_image=st.file_uploader('Please Upload a Photo')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image=Image.open(uploaded_image)

        features=extract_features(os.path.join('Uploads',uploaded_image.name),model,detector)

        index_pos=recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        # display
        col1, col2 = st.columns(2)

        with col1:
            st.header('Your Image')
            st.image(display_image)
        with col2:
            st.header("You Look Like" + predicted_actor)
            st.image(filenames[index_pos], width=300)
