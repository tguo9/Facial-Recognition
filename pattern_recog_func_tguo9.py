import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_digits
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.cm as cm
import cv2 
import matplotlib.image as mpimg
from sklearn import svm
from sklearn import decomposition
from pdb import set_trace

def interpol_im(im, dim1 = 8, dim2 = 8, plot_new_im = False, cmap = 'binary', grid_off = False):

	if (len(im.shape) == 3):
		im = im[:,:,0]

	x = np.arange(im.shape[1])
	y = np.arange(im.shape[0])

	f2d = interp2d(x,y,im)

	x_new = np.linspace(0, im.shape[1], dim1)
	y_new = np.linspace(0, im.shape[0], dim2)

	im_new = f2d(x_new,y_new)

	if (plot_new_im):

		plt.imshow(im_new,interpolation='nearest',cmap=cmap)
		if (grid_off):
			plt.grid('off')
		plt.show()

	return im_new.flatten()

def pca_svm_pred(imfile, md_pca, md_clf, dim1 = 45, dim2 = 60):

	flatten_im = interpol_im(imfile, dim1 = dim1, dim2 = dim2)

	flatten_im = flatten_im.reshape(1,-1)

	flatten_im_proj = md_pca.transform(flatten_im)

	predict = md_clf.predict(flatten_im_proj)

	return predict[0]

def pca_X(X, n_comp = 10):

	md_pca = PCA(n_comp,whiten = True)
	md_pca.fit(X)
	X_proj = md_pca.transform(X)

	return md_pca, X_proj

def rescale_pixel(unseen):

	im_flat = interpol_im(unseen)

	im_flat = im_flat * 15
	im = im_flat.astype(int)

	im = 15 - im

	return im

def svm_train(X, y, gamma = 0.001, C = 100):

	md_clf = svm.SVC(gamma = gamma, C = C)
	md_clf.fit(X, y)

	return md_clf

