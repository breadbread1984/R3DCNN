import numpy as np;
import cv2;
import tensorflow as tf;
from train_r3dcnn import action_model_fn;

class ActionRecognition:
	def __init__(self):
		self.buf = list();
		self.sequence_lengths = np.full((1),10).astype(np.int32);
		self.gesture_classifier = tf.estimator.Estimator(model_fn = action_model_fn, model_dir = "gesture_classifier_model");

	def predict(self,frame):
		assert len(frame.shape) == 3;
		assert frame.shape[0] == 120;
		assert frame.shape[1] == 160;
		assert frame.shape[2] == 3;

		if len(self.buf) == 80:
			self.buf.pop(0);
			self.buf.append(frame[4:116,24:136]);
		elif len(self.buf) < 80:
			self.buf.append(frame[4:116,24:136]);
			return [0];
		else:
			raise Exception('buffer size is over 80');

		features = np.array(self.buf);
		features = features.reshape(1,10,8,112,112,3).astype(np.float32)
		input_fn = lambda:{'data':tf.convert_to_tensor(features),'sequence_lengths':tf.convert_to_tensor(self.sequence_lengths)};
		predictions = self.gesture_classifier.predict(input_fn);
		prediction = next(predictions);
		return prediction;

if __name__ == "__main__":
	cap = cv2.VideoCapture(-1);
	ar = ActionRecognition();
	while True:
		ret, img = cap.read();
		if ret == False: break;
		frame = cv2.resize(img,(160,120));
		sequence = ar.predict(frame);
		for i in range(0,len(sequence)):
			cv2.putText(img,str(sequence[i]),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2);
		cv2.imshow('',img);
		cv2.waitKey(10);
