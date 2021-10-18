from flask import Flask,render_template,request
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import main as m

model=load_model("players.h5")
app=Flask(__name__)

def model_predict(img):
    test_image=image.load_img(img,target_size=(150,150))

    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)


    result=model.predict(test_image)
    for key in m.train_set.items():
        if key[1]==np.argmax(result):
            return key[0]



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