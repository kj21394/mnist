import cv2
import numpy as np

def cnn_cam_mnist(model):
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        cv2.rectangle(frame, (400, 50), (600, 250), (100, 50, 200), 5)
        roi = frame[50:250, 400:600, :]
        roi_p = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_p = cv2.flip(roi_p, 1)
        roi_p = cv2.adaptiveThreshold(roi_p, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
        roi_p = cv2.resize(roi_p, (28,28))
        cv2.imshow('roi',roi_p)
        roi_p = np.reshape(roi_p,(1,28,28,1))
        predict = model.predict(roi_p)
        num = np.count_nonzero(roi_p)
        if (110<num<500):
            text = "Predict:" + str(np.argmax(predict)) 
        else:
            text = "Predict:"
        cv2.putText(frame, text, (5, 420), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == ord('q'):
            break
        # clearing the board
    cap.release()
    cv2.destroyAllWindows()