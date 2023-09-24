from keras.models import load_model
model=load_model('my_model.h5')
global result
import numpy as np
from keras.utils import load_img,img_to_array
def pred(str1):
    test_image=load_img(str1,target_size=(224,224))
    test_image=img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return result
