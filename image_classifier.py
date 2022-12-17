import numpy as np
from tensorflow import keras
import streamlit as st
import cv2
from PIL import Image,ImageOps
import streamlit as st
import base64

classes=['dew', 'fogsmog', 'lightning', 'rain', 'rainbow', 'snow']


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('pexels-miguel-á-padriñán-19670.jpg') 


# st.title("WEATHER CLASSIFICATION USING IMAGE DATA")

@st.cache(allow_output_mutation=True)
def load_model():
  model=keras.models.load_model('weather_model.h5')
  return model


with st.spinner('Model is being loaded..'):
  model=load_model()



st.write("""
         # Weather Classification
         """
         )



file = st.file_uploader("Please upload animage file", type=["jpg"])
st.set_option('deprecation.showfileUploaderEncoding', False)



def import_and_predict(image_data, model):
    
    # image=cv2.imread(image_path)
    # image_resized= cv2.resize(image, (180,180))
    # # rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    # image=np.expand_dims(image_resized,axis=0)


    size = (180,180)    

    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)

    image = np.asarray(image)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_reshape=np.expand_dims(img,axis=0)

    pred=model.predict(img_reshape)
          
    return pred


if file is None:
    st.text("Please upload an image file")


else:
    # st.write(file,file.name)
    image1 = Image.open(file)
    st.image(image1, use_column_width=True)
    predictions = import_and_predict(image1, model)
    output_class=classes[np.argmax(predictions)]
    
    # st.write(""" ## The predicted class is """ ,output_class)
    st.success(f"The  Predicted  Class  is  ''{output_class.upper()}''")
