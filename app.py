from flask import Flask,render_template,request,redirect,url_for
from keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
import os
import numpy as np
import pickle

model=load_model("player_model.h5")
app=Flask(__name__)

def model_predict(img):
    player_lst=pickle.load(open("player_encoder.pkl",'rb'))
    test_image=image.load_img(img,target_size=(150,150))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)

    return player_lst.inverse_transform([np.argmax(model.predict(test_image))])[0]


@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        target_img = os.path.join(os.getcwd() , 'uploads')
            # Get the file from post request
        f = request.files['my_image']

        # Save the file to ./uploads
        f.save(os.path.join(target_img , f.filename))
        img_path = os.path.join(target_img , f.filename)

        # Make prediction
        pred = model_predict(img_path)
        print(img_path)
        return render_template("index.html",prediction=pred,img_path=img_path)

if __name__=="__main__":
    app.run(debug=True)