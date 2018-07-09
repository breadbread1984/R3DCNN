#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import;
from __future__ import division;
from __future__ import print_function;

import os;
import numpy as np;
import tensorflow as tf;

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

batch_size = 4;
class_num = 25;

def main(unused_argv):
	gesture_classifier = tf.estimator.Estimator(model_fn = action_model_fn, model_dir = "gesture_classifier_model");
	tf.logging.set_verbosity(tf.logging.DEBUG);
	logging_hook = tf.train.LoggingTensorHook(tensors = {"loss":"loss"}, every_n_iter = 1);
	gesture_classifier.train(input_fn = train_input_fn,steps = 200000,hooks = [logging_hook]);
	eval_results = gesture_classifier.evaluate(input_fn = eval_input_fn,steps = 1);
	print(eval_results);

def parse_function(serialized_example):
	feature = tf.parse_single_example(
		serialized_example,
		features = {
			'clips': tf.FixedLenFeature((),dtype = tf.string, default_value = ''),
			'label': tf.FixedLenFeature((),dtype = tf.int64, default_value = 0)
		}
	);
	clips = tf.decode_raw(feature['clips'],out_type = tf.uint8);
	clips = tf.reshape(clips,[10,8,112,112,3]);
	clips = tf.cast(clips, dtype = tf.float32);
	sequence_length = tf.constant(clips.get_shape().as_list()[0],dtype = tf.int32);
	label = tf.cast(feature['label'], dtype = tf.int32);
	label = tf.reshape(label,[1]);
	idx = tf.where(tf.not_equal(label,-1));
	label = tf.SparseTensor(indices = idx, values = tf.gather_nd(label,idx), dense_shape = tf.cast(label.get_shape(),dtype = tf.int64));
	return dict(zip(['data','sequence_lengths'],[clips,sequence_length])),label;

def train_input_fn():
	dataset = tf.data.TFRecordDataset(['trainset.tfrecord']);
	dataset = dataset.map(parse_function);
	dataset = dataset.shuffle(buffer_size = 512);
	dataset = dataset.batch(batch_size);
	dataset = dataset.repeat(None);
	iterator = dataset.make_one_shot_iterator();
	features, labels = iterator.get_next();
	return features, labels;

def eval_input_fn():
	dataset = tf.data.TFRecordDataset(['testset.tfrecord']);
	dataset = dataset.map(parse_function);
	dataset = dataset.shuffle(buffer_size = 512);
	dataset = dataset.batch(batch_size);
	dataset = dataset.repeat(None);
	iterator = dataset.make_one_shot_iterator();
	features, labels = iterator.get_next();
	return features, labels;

def action_model_fn(features, labels, mode):
	#data.shape = [batch_size = ?, time_steps = 10, depth = 8, height = 112, width = 112, channel = 3]
	#labels.shape = [batch_size = ?,label_length = 1]
	data = features["data"];
	sequence_lengths = features["sequence_lengths"];

	timesteps = data.get_shape().as_list()[1];

	#create lstm operator object
	stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(512) for _ in range(1)]);

	conv_input = tf.reshape(data,[-1,8,112,112,3]);
	#layer 1
	c1 = tf.layers.conv3d(conv_input,filters = 64, kernel_size = [3,3,3], padding = "same");
	b1 = tf.contrib.layers.layer_norm(c1,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
	p1 = tf.layers.max_pooling3d(b1,pool_size = [1,2,2], strides = [1,2,2], padding = "same");
	#layer 2
	c2 = tf.layers.conv3d(p1,filters = 128, kernel_size = [3,3,3], padding = "same");
	b2 = tf.contrib.layers.layer_norm(c2,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
	p2 = tf.layers.max_pooling3d(b2,pool_size = [2,2,2], strides = [2,2,2], padding = "same");
	#layer 3
	c3a = tf.layers.conv3d(p2,filters = 256, kernel_size = [3,3,3], padding = "same");
	b3a = tf.contrib.layers.layer_norm(c3a,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
	c3b = tf.layers.conv3d(b3a,filters = 256, kernel_size = [3,3,3], padding = "same");
	b3b = tf.contrib.layers.layer_norm(c3b,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
	p3 = tf.layers.max_pooling3d(b3b,pool_size = [2,2,2], strides = [2,2,2], padding = "same");
	#layer 4
	c4a = tf.layers.conv3d(p3,filters = 512, kernel_size = [3,3,3], padding = "same");
	b4a = tf.contrib.layers.layer_norm(c4a,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
	c4b = tf.layers.conv3d(b4a,filters = 512, kernel_size = [3,3,3], padding = "same");
	b4b = tf.contrib.layers.layer_norm(c4b,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
	p4 = tf.layers.max_pooling3d(b4b,pool_size = [2,2,2], strides = [2,2,2], padding = "same");
	#layer 5
	c5a = tf.layers.conv3d(p4,filters = 512, kernel_size = [3,3,3], padding = "same");
	b5a = tf.contrib.layers.layer_norm(c5a,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
	c5b = tf.layers.conv3d(b5a,filters = 512, kernel_size = [3,3,3], padding = "same");
	b5b = tf.contrib.layers.layer_norm(c5b,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
	#load c3d pretrained parameters
	tf.train.init_from_checkpoint("action_classifier_model",{v.name.split(':')[0]: v for v in tf.contrib.framework.get_variables_to_restore()});
	#the last pooling layer of C3D is removed because the input clips length become 8 frames
	#r5b.shape = [batch_size * time_steps = ?*10, depth = 1, height = 7, width = 7, channel = 512]
	f = tf.layers.flatten(b5b);
	d1 = tf.layers.dense(f,units = 512, activation = tf.nn.relu);
	dp1 = tf.layers.dropout(d1,training = mode == tf.estimator.ModeKeys.TRAIN);
	d2 = tf.layers.dense(dp1,units = 512, activation = tf.nn.relu);
	dp2 = tf.layers.dropout(d2,training = mode == tf.estimator.ModeKeys.TRAIN);
	#dp2.shape = [batch_size = ?, time_steps = 10, feature_dim = 512]
	lstm_input = tf.reshape(dp2,[-1,timesteps,512])
	output, _ = tf.nn.dynamic_rnn(stacked_lstm, lstm_input, sequence_length = sequence_lengths, time_major = False, dtype = tf.float32);
	output = tf.reshape(output,[-1,512]);
	#logits.shape = [batch_size * time_steps = ?*10, num_classes]
	logits = tf.layers.dense(output,units = class_num);
	#logits.shape = [batch_size = ?, times_steps = 10, num_classes]
	logits = tf.reshape(logits,[-1,timesteps,class_num]);
	#predict mode
	if mode == tf.estimator.ModeKeys.PREDICT:
		#logits.shape = [times_steps = 10, batch_size = ?, num_classes]
		logits = tf.transpose(logits,[1,0,2]);
		decoded,_ = tf.nn.ctc_beam_search_decoder(logits,sequence_lengths);
		tf.Print(decoded[0].dense_shape,[decoded[0].dense_shape],message = "shape = ");
		predictions = tf.sparse_tensor_to_dense(decoded[0]);
		return tf.estimator.EstimatorSpec(mode = mode,predictions = predictions);
	#train mode
	if mode == tf.estimator.ModeKeys.TRAIN:
		loss = tf.nn.ctc_loss(labels,logits,sequence_lengths,time_major = False);
		loss = tf.reduce_mean(loss,name = "loss");
		optimizer = tf.train.AdamOptimizer(1e-4);
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step());
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op);
	#eval mode
	if mode == tf.estimator.ModeKeys.EVAL:
		loss = tf.nn.ctc_loss(labels,logits,sequence_lengths,time_major = False);
		loss = tf.reduce_mean(loss,name = "loss");
		#logits.shape = [times_steps = 10, batch_size = ?, num_classes]
		logits = tf.transpose(logits,[1,0,2]);
		decoded,_ = tf.nn.ctc_beam_search_decoder(logits,sequence_lengths);
		eval_metric_ops = {"mean_edit_distance": tf.metrics.mean(tf.edit_distance(tf.cast(decoded[0],tf.int32),labels))};
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops);

	raise Exception('Unknown mode of estimator!');

if __name__ == "__main__":
	tf.app.run();
