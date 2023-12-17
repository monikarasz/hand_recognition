import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import keyboard
import time

cap=cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pos_df = pd.DataFrame(columns= np.reshape(np.array([[str(i)+'x' for i in range(21)],[str(i)+'y' for i in range(21)]]),42,order='F'))
class handTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):#zmieniłam max liczbę rąk na 1
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 10, (0, 0, 0), cv2.FILLED) #kółko na końcu małego palca
        return lmlist
def main():
    print("Please show your palm to the camera and move your hand and fingers in all directions until the video stops")
    print("After that you will be able to chose if you want to add any other users")
    global pos_df
    user_count = 1
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    i=0
    while True:
        while i<200:
            success,image = cap.read()#image is an ndarray
            image = tracker.handsFinder(image)#still an ndarray
            lmList = tracker.positionFinder(image)

            if len(lmList) == 21:
                lmList_flat = np.reshape(np.array([np.array(lmList)[:, 1], np.array(lmList)[:, 2]]), 42, order='F')
                pos_df.loc[len(pos_df.index)] = lmList_flat.tolist()
                i += 1
            cv2.imshow("Video",image)
            cv2.waitKey(25)#I changed it from 1 to 250

            
        print(pos_df)
        fname = "user_" + str(user_count) +".csv"
        pos_df.to_csv(fname,index = False) #nazwa pliku uzalezniona od user count

        print ("press n to add more data")
        print ("press q to quit")

        opt = input()
        if opt == 'n':
            user_count += 1
            print('next user in 3s...')
            i=0
            pos_df = pd.DataFrame(columns=np.reshape(np.array([[str(i) + 'x' for i in range(21)], [str(i) + 'y' for i in range(21)]]), 42, order='F'))
            time.sleep(3) #usypia na 3s
        if opt == 'q':
            print('quitting...')
            break
        else:
            print('unrecognizable option. quitting...')

if __name__ == "__main__":
    main()
