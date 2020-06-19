# CommaAI
Comma AI Speed Test Challenge

Goal : Predict the speed of teh car from the video

_Process_
1) Pass all the arguments and initialize the parameters
2) Create dictionary for Lucas Kanade Optical flow calculations: Tracking and Extracting features initiazlized. Load training speed per frame values from text file.
3) Construct a mask on the frame, to focus on road for speed calculations.
4) Convert Color Frame -> Gray -> Gaussian BLur removing noise
5) Calculate optical Flow for every frame by using Shi-Tomasi Corner detection
6) Create a window of visualization of good features.
7) Split the data into training and validation and calculate moving average for both datasets.
8) Fir a linear regression model and calculate Mean Square error, hf factor.
9) Finally from the hf factor generated we predict speef for the test video and save it in a text file.

![](https://github.com/aayushi-95/CommaAI/blob/master/images/Figure_1.PNG) ![](https://github.com/aayushi-95/CommaAI/blob/master/images/Figure_1-1.PNG)

![](https://github.com/aayushi-95/CommaAI/blob/master/images/Capt1ure.PNG)
