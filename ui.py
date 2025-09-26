import streamlit as st
import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
img2vec=Img2Vec()

model=pickle.load(open('model.p','rb'))

st.title('Feature Extraction from Image data')
st.set_page_config(layout='wide')


upload_img=st.file_uploader('upload any image',type=['png','jpeg','jpg'])
if upload_img is not None:
        img=Image.open(upload_img).convert('RGB')
        st.image(img)
        feat=img2vec.get_vec(img)

        pred=model.predict([feat])
        st.write(pred)
else:
    st.error('Please select any input_mode')

