import keras
import numpy as np
from scipy import linalg as LA



def wct(fc, fs):
	print fc.shape
	fc = fc.reshape(512*49)
	fc = fc - fc.mean() #Taking mean along axis = 1
	fc = fc.reshape((fc.size, 1))
	fc_covariance_mat = np.matmul(fc, fc.transpose())
	c_u, c_e, c_v = LA.svd((fc_covariance_mat))

	print c_e.shape, c_e.size
	for i in range(c_e.size):
		print i,
		if(c_e[i] < 0.00001):
			end_c = i
			break

	print "DONE1"
	c_d = np.power(c_e[0:end_c], -0.5)
	whiten = np.matmul( np.matmul(c_v[:,0:end_c], np.diag(c_d)), c_v[:,0:end_c].transpose() )
	fc_cap = np.matmul(whiten, fc)
	#Should add mean of fc to fc

	print "DONE2"
	fs = fs.reshape( 512*49 )
	style_mean = fs.mean() #Taking mean along axis = 1
	fs = fs - style_mean
	fs = fs.reshape(fs.size, 1)
	fs_covariance_mat = np.matmul(fs, fs.transpose())
	s_u, s_e, s_v = LA.svd((fs_covariance_mat))

	print "DONE3"
	for i in range(s_e.shape):
		if(s_e[i] < 0.00001):
	 		end_s = i
	 		break

	s_d = np.power( s_e[0:end_s], 0.5 )
	coloring = np.matmul(np.matmul( np.matmul(s_v[:,0:end_s], np.diag(s_d)), s_v[:,0:end_s].transpose()), fc_cap)
	target = coloring + style_mean
	print "DONE"
	return target
