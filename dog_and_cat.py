import tensorflow as tf
import cv2
from keras.models import load_model
import numpy as np

model = load_model("model\dogs_and_cats_classification.h5", compile=False)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['accuracy'])

font = cv2.FONT_HERSHEY_COMPLEX
org = (50, 100)
color = (0, 0, 0)

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    frame_for_predict = cv2.resize(frame, dsize=(160, 160))
    frame_for_predict = tf.expand_dims(frame_for_predict, axis=0)

    if status:
        logits = model.predict_on_batch(frame_for_predict)
        predict_per = tf.nn.sigmoid(logits)

        predict = tf.where(predict_per < 0.5, 0, 1)

        if 0.25 < predict_per < 0.75:
            text = "I don't know??"
            cv2.putText(frame, text, org, font, 1, color, 2)
        else:
            if predict == 1:
                text = "dog = %.2f" % (predict_per * 100)
                cv2.putText(frame, text, org, font, 1, color, 2)
            elif predict == 0:
                text = "cat = %.2f" % ((1 - predict_per) * 100)
                cv2.putText(frame, text, org, font, 1, color, 2)

        cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()