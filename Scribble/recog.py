from keras.models import load_model
import numpy as np
from PIL import Image

def test():
    input_image = Image.open('number_input.png').convert('L')
    input_image = input_image.resize((28,28))
    img = np.expand_dims(input_image, axis=-1)
    try:
        data = np.asarray( img, dtype='uint8' )
    except SystemError:
        data = np.asarray( img.getdata(), dtype='uint8' )


    model = load_model('mnist.h5')

    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    pred = model.predict(data.reshape(1, 28, 28, 1))
    #print('Predicted Digit : ' + str(pred))
    x = str(pred.argmax())
    return x