from keras.models import load_model
import matplotlib.pyplot as plt
model=load_model("players.h5")

train_set={'Bhuvneshwar Kumar': 0,
 'Dinesh Karthik': 1,
 'Hardik Pandya': 2,
 'Jasprit Bumrah': 3,
 'KL Rahul': 4,
 'Mohammed Shami': 5,
 'Ms Dhoni': 6,
 'Ravindra Jadeja': 7,
 'Rohit Sharma': 8,
 'Shikhar Dhawan': 9,
 'Virat Kohli': 10,
 'Yuzvendra Chahal': 11}


def predict(image):
    import numpy as np
    from keras.preprocessing import image
    test_image=image.load_img(image,target_size=(150,150))
    plt.imshow(test_image)
    print(plt.show())
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)

    result=model.predict(test_image)
    for key in train_set.items():
        if key[1]==np.argmax(result):
            return key[0]