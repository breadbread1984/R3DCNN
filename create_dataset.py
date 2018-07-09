#!/usr/bin/python 
# -*- coding: utf-8 -*-

import cv2;
import numpy as np;
from numpy.random import uniform;
import math;
import os.path;
import tensorflow as tf;
import readdata;

def selectCropCenter(frame_size):
	assert frame_size[0] >= 112;
	assert frame_size[1] >= 112;
	x_min = 112 // 2;
	x_max = frame_size[0] - x_min;
	y_min = 112 // 2;
	y_max = frame_size[1] - y_min;
	return (uniform(x_min,x_max),uniform(y_min,y_max));

def rotateImage(image,center,angle,scale):
	assert image is not None;
	rot_mat = cv2.getRotationMatrix2D(center,angle,scale);
	result = cv2.warpAffine(image,rot_mat,(112,112),flags=cv2.INTER_LINEAR);
	result = cv2.resize(result,(112,112));
	return result;

def video2sample():
	sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
	file_lists = dict()
	file_lists["test"] = "./nvgesture_test_correct_cvpr2016_v2.lst"
	file_lists["train"] = "./nvgesture_train_correct_cvpr2016_v2.lst"
	train_list = list()
	test_list = list()

	readdata.load_split_nvgesture(file_with_split = file_lists["train"],list_split = train_list)
	readdata.load_split_nvgesture(file_with_split = file_lists["test"],list_split = test_list)
	
	#trainset
	if True == os.path.exists('trainset.tfrecord'):
		os.remove('trainset.tfrecord');
	writer = tf.python_io.TFRecordWriter('trainset.tfrecord');
	for trainIdx in range(0,len(train_list)):
		#提取前景样本
		data,label = readdata.load_data_from_file(example_config = train_list[trainIdx],sensor = sensors[0], image_width = 160, image_height = 120);
		#transpose to [frame_num = 80,image_height,image_width,chnum]
		data = np.transpose(data,[3,0,1,2]);
		#sample 5 cropped areas from video
		for sampleIdx in range(0,5):
			crop_center = selectCropCenter((160,120));
			rotate_angle = uniform(-15,15);
			scale = uniform(0.8,1.2);
			features = np.zeros((80,112,112,3),dtype = np.uint8);
			for frameIdx in range(0,80):
				features[frameIdx,...] = rotateImage(data[frameIdx,...],crop_center,rotate_angle,scale);
			features = np.reshape(features,[10,8,112,112,3]);
			trainsample = tf.train.Example(features = tf.train.Features(
				feature = {
					'clips': tf.train.Feature(bytes_list = tf.train.BytesList(value = [features.tobytes()])),
					'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
				}
			));
			writer.write(trainsample.SerializeToString());
	writer.close();

	#testset
	if True == os.path.exists('testset.tfrecord'):
		os.remove('testset.tfrecord');
	writer = tf.python_io.TFRecordWriter('testset.tfrecord');
	for testIdx in range(0,len(test_list)):
		data,label = readdata.load_data_from_file(example_config = test_list[testIdx],sensor = sensors[0], image_width = 160, image_height = 120);
		#transpose to [frame_num = 80,image_height,image_width,chnum]
		data = np.transpose(data,[3,0,1,2]);
		#sample 5 cropped areas from video
		for sampleIdx in range(0,5):
			crop_center = selectCropCenter((160,120));
			rotate_angle = uniform(-15,15);
			scale = uniform(0.8,1.2);
			features = np.zeros((80,112,112,3),dtype = np.uint8);
			for frameIdx in range(0,80):
				features[frameIdx,...] = rotateImage(data[frameIdx,...],crop_center,rotate_angle,scale);
			features = np.reshape(features,[10,8,112,112,3]);
			trainsample = tf.train.Example(features = tf.train.Features(
				feature = {
					'clips': tf.train.Feature(bytes_list = tf.train.BytesList(value = [features.tobytes()])),
					'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
				}
			));
			writer.write(trainsample.SerializeToString());
	writer.close();

if __name__ == "__main__":
	video2sample();
