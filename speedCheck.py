import cv2
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt

class SpeedCheck():
	def __init__(self, config):
		self.vid = cv2.VideoCapture(config.trainvid)
		self.txtfile = config.traintxt
		self.vis = config.visual
		self.len_gt = config.gt_len
		self.test_vid = cv2.VideoCapture(config.testvid)
		self.setupVal()

	def setupVal(self):
		self.lk_params = dict(winSize = (21, 21),
							  maxLevel = 2,
							  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

		self.frame_idx = 0
		self.prev_pts = None
		self.detect_interval = 1
		self.temp_preds = np.zeros(int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)))

		with open(self.txtfile, 'r') as file_:
			gt = file_.readlines()
			gt = [float(x.strip()) for x in gt]
		
		self.gt = np.array(gt[:self.len_gt])

		self.window = 80 #Moving average window set to 80
		self.prev_gray = None
        
        #Lucas Kanade parameters initialized

	def constructMask(self, mask = None, test=True):
		vid = self.test_vid if test else self.vid
		
		if mask is None:
			W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
			H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
			mask = np.zeros(shape = (H,W), dtype = np.uint8)
			mask.fill(255)
		else:
			W = mask.shape[1]
			H = mask.shape[0]

		cv2.rectangle(mask, (0, 0), (W, H), (0, 0, 0), -1)

		x_top_offset = 180 
		x_btm_offset = 50

		poly_pts = np.array([[[640-x_top_offset, 250], [x_top_offset, 250], [x_btm_offset, 350], [640-x_btm_offset, 350]]], dtype=np.int32)
		cv2.fillPoly(mask, poly_pts, (255, 255, 255))

		return mask


	def FrameCal(self, frame):
		frame = cv2.GaussianBlur(frame, (3,3), 0)

		curr_pts, _st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame, self.prev_pts, None, **self.lk_params)
		flow = np.hstack((self.prev_pts.reshape(-1, 2), (curr_pts - self.prev_pts).reshape(-1, 2)))

		preds = []
		for x, y, u, v in flow:
			if v < -0.05:
				continue
			x -= frame.shape[1]/2
			y -= frame.shape[0]/2

			if y == 0 or (abs(u) - abs(v)) > 11:
				preds.append(0)
				preds.append(0)
			elif x == 0:
				preds.append(0)
				preds.append(v / (y*y))
			else:
				preds.append(u / (x * y))
				preds.append(v / (y*y))

		return [n for n in preds if n>=0]
    
    #Considering median points and balancing the stability 

	def getKeyPts(self, offset_x=0, offset_y=0):
		if self.prev_pts is None:
		  return None
		return [cv2.KeyPoint(x=p[0][0] + offset_x, y=p[0][1] + offset_y, _size=10) for p in self.prev_pts]
    
	def computeAverage(self, x, window,idx):
		min_idx = max(0, idx - window - 1)
		return np.mean(x[min_idx:idx])
	
	def getFeatures(self, frame_gray, mask):
		return cv2.goodFeaturesToTrack(frame_gray,30,0.1,10,blockSize=10,
													mask=mask)
#Extract Harris corner key features and return with offset
													

#Main driver function
	def run(self):

		mask = self.constructMask()
		prev_key_pts = None

		while self.vid.isOpened() and self.frame_idx<len(self.gt):
			ret, frame = self.vid.read()
			if not ret:
				break

			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame_gray = frame_gray[130:350, 35:605]
			vmask = frame.copy() 
			
		
			if self.prev_pts is None:
				self.temp_preds[self.frame_idx] = 0 #Each frame is processed and stored in an array of respective frame index
			else:
				preds = self.FrameCal(frame_gray)
				self.temp_preds[self.frame_idx] = np.median(preds) if len(preds) else 0

			self.prev_pts = self.getFeatures(frame_gray, mask[130:350, 35:605])
			self.prev_gray = frame_gray
			self.frame_idx += 1 #FEature Extraction
			
			if self.vis:
				prev_key_pts = self.visualize(frame, vmask, prev_key_pts)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

		
		self.vid.release()

		# Split data into train and validation - 
		split = self.frame_idx//10
		train_preds = self.temp_preds[:self.frame_idx-split]
		val_preds = self.temp_preds[self.frame_idx - split:self.frame_idx]
		gt_train = self.gt[:len(train_preds)]
		gt_val = self.gt[len(train_preds):self.frame_idx]

		preds = self.movingAverage(train_preds,self.window) #Fit ground truth labels

		lin_reg = linear_model.LinearRegression(fit_intercept=False)
		lin_reg.fit(preds.reshape(-1, 1), gt_train) 
		hf_factor = lin_reg.coef_[0]

		pred_speed_train = train_preds * hf_factor
		pred_speed_train = self.movingAverage(pred_speed_train,self.window)
		mse = np.mean((pred_speed_train - gt_train)**2)
		print("MSE calculated for trainining data", mse)

		pred_speed_val = val_preds * hf_factor
		pred_speed_val = self.movingAverage(pred_speed_val,self.window)
		mse = np.mean((pred_speed_val - gt_val)**2)
		print("MSE calculated for validate data", mse)
		
		self.plot(pred_speed_train, gt_train)
		self.plot(pred_speed_val, gt_val)

		return hf_factor

	def visualize(self, frame, vmask, prev_key_pts, speed=None):
		self.constructMask(vmask)
		vmask = cv2.bitwise_not(vmask)
		frame_vis = cv2.addWeighted(frame, 1, vmask, 0.3, 0)
		key_pts = self.getKeyPts(35, 130)
		cv2.drawKeypoints(frame_vis, key_pts, frame_vis, color=(0,0,255))
		cv2.drawKeypoints(frame_vis, prev_key_pts, frame_vis, color=(0,255,0))

		if speed is not None:
			font = cv2.FONT_ITALIC
			cv2.putText(frame_vis, "speed{}".format(speed), (10, 35), font, 1.2, (0, 0, 255))
		cv2.imshow('test',frame_vis)

		return key_pts
    
	def plot(self, predict_values,gt):
		fig,ax = plt.subplots()
		ax.plot(np.arange(len(gt)), gt, label='Val', color = 'coral')
		ax.plot(np.arange(len(predict_values)), np.array(predict_values), label='Prediction',color='c')
		start,end =ax.get_xlim()
		ax.legend(loc='upper right')
		plt.xlabel('Frame Number')
		plt.ylabel('Speed Estimation')        

		plt.show()
    
	def movingAverage(self, x, window):
		
		ret = np.zeros_like(x)

		for i in range(len(x)):
			idx1 = max(0, i - (window - 1) // 2)
			idx2 = min(len(x), i + (window - 1) // 2 + (2 - (window % 2)))
			ret[i] = np.mean(x[idx1:idx2])

		return ret

#Process each frame -> Extract features -> Calculate MSE -> Fit the labels to our model -> Visualize
	def test(self, hf_factor, save_test_txt=False):
		mask = self.constructMask(test=True)
		
		self.prev_gray = None
		test_preds = np.zeros(int(self.test_vid.get(cv2.CAP_PROP_FRAME_COUNT)))
		frame_idx = 0
		prev_key_pts = None
		self.prev_pts = None
		
		while self.test_vid.isOpened():
			ret, frame = self.test_vid.read()
			if not ret:
				break

			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			frame_gray = frame_gray[130:350, 35:605]
			vmask = frame.copy() #
			
			pred_speed = 0
			if self.prev_pts is None:
				test_preds[frame_idx] = 0
			else:
				preds = self.FrameCal(frame_gray)
				pred_speed = np.median(preds) * hf_factor if len(preds) else 0
				test_preds[frame_idx] =  pred_speed

		
			self.prev_pts = self.getFeatures(frame_gray, mask[130:350, 35:605])
			self.prev_gray = frame_gray
			frame_idx += 1
			
			vis_pred_speed = self.computeAverage(test_preds, self.window//2, frame_idx)
			prev_key_pts = self.visualize(frame, vmask, prev_key_pts, speed=vis_pred_speed)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		self.test_vid.release()
        
		
		print("Saving predicted speeds in test.txt ")
		if save_test_txt:
			with open(r"data/test.txt", "w+") as file_:
				for item in test_preds:
					file_.write("%s \n" % item)
                    
    
                    





