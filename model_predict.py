# Load saved model
import kashgari

loaded_model = kashgari.utils.load_model('time_ner.h5')

while True:
    text = input('sentence: ')
    t = loaded_model.predict([[char for char in text]])
    print(t)