import keras
import numpy as np
from numpy import linalg as LA



def wct(fc, fs):
	print fc.shape
	fc = fc - fc.mean(1).mean(1)[:,None,None] #Taking mean along axis = 1
	c_u, c_e, c_v = LA.svd((fc.dot(fc.transpose())))

	for i in range(fc.shape[0]):
		if(c_e[i] < 0.00001):
			end_c = i
			break

	c_d = (c_e[0:k_c]).power(-0.5)
	whiten = c_v[:,0:end_c]*np.diag(c_d)*c_v[:,0:end_c].transpose()
	fc_cap = whiten * fc

	# style_mean = fs.mean(1) #Taking mean along axis = 1
	# fs = fs - style_mean
	# s_u, s_e, s_v = LA.svd((fs*fs.transpose())/(fs.shape[0]-1))

	# for i in range(fs.shape[0]):
	# 	if(s_e[i] < 0.00001):
	# 		end_s = i
	# 		break

	# s_d = (s_e[0:k_s]).power(0.5)
	# coloring = s_v[:,0:k_s]*np.diag(s_d)*s_v[:,0:k_s].transpose()*fc_cap
	# target = coloring + style_mean
	# return target	



