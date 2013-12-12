# CS205 project
# image_analogies_serial.py
# Fangzhou Yu, Jessica Yao

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import time
import colorsys
import scipy.misc

# create kernel weights 
kernel5 = np.array([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.

# ANN search function
def ANNalgo(flnn,n,hh,ww,feature_vectorA_prime,f_B_reshape):
	# find the corresponding feature vector for the layer
	f_B = np.array(f_B_reshape[n],dtype=np.float32)

	hA,wA = feature_vectorA_prime[n].shape[0:2]

	# find the ANN!
	print "be patient, doing ANN takes a while"
	indices,dist = flnn.knnSearch(f_B,1,params={})
	print "done!"

	# convert the index to row and col
	jminlist = indices % wA
	iminlist = indices/wA

	result = np.zeros([hh,ww],dtype=np.float32)

	for i in xrange(hh):
		for j in xrange(ww):
			if i > 2 and j > 2 and i < hh-2 and j < ww-2:
				k = i*ww+j

	imin = iminlist[k,0]
	jmin = jminlist[k,0]

	result[i,j] = feature_vectorA_prime[n][imin,jmin][12]

	return iminlist,jminlist,result

# brute force search function
def brutealgo(n,i,j,feature_vectorA,feature_vectorA_prime,feature_vectorB):
	# find the corresponding feature vector for i,j
	f_B = feature_vectorB[n][i,j]

	h,w = feature_vectorA[n].shape[0:2]

	minn = float('inf')
	imin = 0
	jmin = 0

	for y in xrange(h):
		for x in xrange(w):
			diff = np.linalg.norm(kernel5*(feature_vectorA[n][y,x] - f_B))

			if diff < minn:
				minn = diff
				imin,jmin = y,x

	result = feature_vectorA_prime[n][imin,jmin][12]


	return result

if __name__ == '__main__':

	argv = sys.argv

	if len(argv) != 5:
		print "Usage: python", argv[0], "[imageA] [imageA'] [imageB] [output_file]"
		exit()

	# Read image files
	# setup A, A', B, B'
	A_name = (argv[1])
	A_prime_name = (argv[2])
	B_name = (argv[3])
	B_prime_name = argv[4]

	# Read image 3 channels
	imageA = scipy.misc.imread(A_name)
	imageA_prime = scipy.misc.imread(A_prime_name)
	imageB = scipy.misc.imread(B_name)

	# determine the shape of each image
	heightA, widthA = np.int32(imageA.shape[0:2])
	heightB, widthB = np.int32(imageB.shape[0:2])

	print "Processing A %d x %d image" % (heightA,widthA)
	print "Processing B %d x %d image" % (heightB,widthB)

	start_time = time.time()

	# convert images to YIQ
	A_YIQ = np.zeros(imageA.shape, dtype=np.float32)
	A_prime_YIQ = np.zeros(imageA.shape, dtype=np.float32)
	B_YIQ = np.zeros([heightB,widthB,3], dtype=np.float32)
	B_Y = np.zeros(imageB.shape[0:2], dtype=np.float32)
	A_Y = np.zeros(imageA.shape[0:2], dtype=np.float32)
	A_prime_Y = np.zeros(imageA.shape[0:2], dtype=np.float32)

	for i in xrange(heightA):
		for j in xrange(widthA):
			colors = imageA[i,j]/255.
			colorsp = imageA_prime[i,j]/255.
			YIQA = colorsys.rgb_to_yiq(colors[0],colors[1],colors[2])
			YIQA_prime = colorsys.rgb_to_yiq(colorsp[0],colorsp[1],colorsp[2])
			A_YIQ[i,j] = YIQA
			A_Y[i,j] = YIQA[0]
			A_prime_YIQ[i,j] = YIQA_prime 
			A_prime_Y[i,j] = YIQA_prime[0]

	for i in xrange(heightB):
		for j in xrange(widthB):
			colors = imageB[i,j]/255.
			YIQB = colorsys.rgb_to_yiq(colors[0],colors[1],colors[2])
			B_YIQ[i,j] = YIQB
			B_Y[i,j] = YIQB[0]

	stop_time = time.time()
	print "RGB to YIQ conversion complete"
	print "Serial: %f" % (stop_time - start_time)

	# Initialize the image region as empty
	B_prime_YIQ = np.zeros(imageB.shape, dtype=np.float32)
	B_prime_Y = np.zeros(imageB.shape[0:2], dtype=np.float32)
	B_prime_RGB = np.zeros(imageB.shape, dtype=np.float32)

	# create gaussian pyramids 

	start_time = time.time()

	# make a list of pyramids
	AA = A_Y.copy()
	AA_prime = A_prime_Y.copy()
	BB = B_Y.copy()
	pyramidA = [AA]
	pyramidA_prime = [AA_prime]
	pyramidB = [BB]

	for i in xrange(4):
		AA = cv2.pyrDown(AA)
		AA_prime = cv2.pyrDown(AA_prime)
		BB = cv2.pyrDown(BB)

		pyramidA.append(AA)
		pyramidA_prime.append(AA_prime)
		pyramidB.append(BB)

	stop_time = time.time()
	print "pyramids complete"
	print "Serial: %f" % (stop_time - start_time)

	start_time = time.time()

	# create feature vectors for each pixel from coarse to high resolution
	feature_vectorA = []
	for layer in pyramidA:
		#print layer.shape
		htemp, wtemp = np.int32(layer.shape)
		tempstore = np.zeros([htemp,wtemp,25], dtype=np.float32)
		for i in xrange(htemp):
			for j in xrange(wtemp):
				ftemp = np.zeros([25], dtype=np.float32)
				if i > 2 and j > 2 and i < htemp - 2 and j < wtemp - 2:
					index = 0
					for y in xrange(-2,3,1):
						for x in xrange(-2,3,1):
							ftemp[index] = layer[i+y,j+x]
							index = index+1 
					tempstore[i,j] = ftemp      
		feature_vectorA.append(tempstore)

	feature_vectorA_prime = []
	for layer in pyramidA_prime:
		#print layer.shape
		htemp, wtemp = np.int32(layer.shape)
		tempstore = np.zeros([htemp,wtemp,25], dtype=np.float32)
		for i in xrange(htemp):
			for j in xrange(wtemp):
				ftemp = np.zeros([25], dtype=np.float32)
				if i > 2 and j > 2 and i < htemp - 2 and j < wtemp - 2:
					index = 0
					for y in xrange(-2,3,1):
						for x in xrange(-2,3,1):
							ftemp[index] = layer[i+y,j+x]
							index = index+1 
					tempstore[i,j] = ftemp      
		feature_vectorA_prime.append(tempstore)

	feature_vectorB = []
	for layer in pyramidB:
		#print layer.shape
		htemp, wtemp = np.int32(layer.shape)
		tempstore = np.zeros([htemp,wtemp,25], dtype=np.float32)
		for i in xrange(htemp):
			for j in xrange(wtemp):
				ftemp = np.zeros([25], dtype=np.float32)
				if i > 2 and j > 2 and i < htemp - 2 and j < wtemp - 2:
					index = 0
					for y in xrange(-2,3,1):
						for x in xrange(-2,3,1):
							ftemp[index] = layer[i+y,j+x]
							index = index+1 
					tempstore[i,j] = ftemp      
		feature_vectorB.append(tempstore)

	print "feature vectors made"
	stop_time = time.time()
	print "Serial: %f" % (stop_time - start_time)

	start_time = time.time()

	#brute force!!
	#n = 3
	#hh,ww = pyramidB[n].shape
	#print hh,ww
	#temp = np.zeros([hh,ww],dtype=np.float32)
	#for i in xrange(hh):
		#for j in xrange(ww):
			#if i > 2 and j > 2 and i < hh-2 and j < ww-2:
				#temp[i,j] = brutealgo(n,i,j,feature_vectorA,feature_vectorA_prime,feature_vectorB)
			#print i,j

	#print temp
	#result = plt.imshow(temp)
	#plt.show()

	#ANN!!!
	# next we need to flatten our 3D feature vector into 2D for ANN to work
	f_A_reshape = []
	f_B_reshape = []
	for i in xrange(5):
		f_hA,f_wA = feature_vectorA[i].shape[0:2]
		f_A_reshape.append(kernel5*np.array(np.reshape(feature_vectorA[i],[f_hA*f_wA,25]),dtype=np.float32))

		f_hB,f_wB = feature_vectorB[i].shape[0:2]
		f_B_reshape.append(kernel5*np.array(np.reshape(feature_vectorB[i],[f_hB*f_wB,25]),dtype=np.float32))

	# train dataset
	n = 0
	trainset = np.array(f_A_reshape[n],dtype=np.float32)
	params = dict(algorithm=1,trees=4)
	flnn = cv2.flann_Index(trainset,params)

	# now to perform the actual ANN
	hh,ww = pyramidB[n].shape
	ilist,jlist,B_prime_Y = ANNalgo(flnn,n,hh,ww,feature_vectorA_prime,f_B_reshape)

	stop_time = time.time()
	print "algo complete"
	print "Serial: %f" % (stop_time - start_time)

	start_time = time.time()

	# COHERENCE CODE 
	for i in xrange(2, heightB - 1, 1):
		for j in xrange(2, widthB - 1, 1):          
			#set coh distance = infinity
			coh_distance = float("inf")
			#current pixel = match
			match = (i,j)
			#pixel in A that matched this pixel
			k = i * widthB + j
			a_loc = (ilist[k,0],jlist[k,0])
			k_best = 0
			if (a_loc[0] > 1 and a_loc[0] < heightA -2 and a_loc[1] > 1 and a_loc[1] < widthA-2):
				#print i,j
				for i_plus in xrange(-2,1,1):
					for j_plus in xrange(-2, 3, 1):
						#go thru neighborhood and find best match
						#find the k that correspond to each pixel
						k = ((i + i_plus) % heightB) * widthB  + ((j + j_plus) % widthB)
						i_new, j_new = (ilist[k,0],jlist[k,0])            
						d = np.linalg.norm(kernel5*(feature_vectorB[n][i][j] - feature_vectorA[n][i_new][j_new])) 
						if i_plus == 0 and j_plus == 0:
							ann_distance = d
							break                    
						if d < coh_distance:
							coh_distance = d
							match = (i_new, j_new)
							k_best = k
						# print coh_distance, match, ann_distance, a_loc
				if coh_distance < 2*2*ann_distance:
					B_prime_Y[i][j] = feature_vectorA_prime[n][match[0]][match[1]][12]
					ilist[k_best,0] = match[0]
					jlist[k_best,0] = match[1]


	stop_time = time.time()
	print "coherence complete"
	print "Serial: %f" % (stop_time - start_time)

	# conversion back from YIQ to RGB
	start_time = time.time()

	BBB = B_YIQ.copy()
	# B_prime_Y to B_prime_YIQ
	for i in xrange(heightB):
		for j in xrange(widthB):
			temparray = BBB[i,j]
			temparray[0] = B_prime_Y[i,j]
			B_prime_YIQ[i,j] = temparray

	# B_prime_YIQ to B_prime_RGB
	for i in xrange(heightB):
		for j in xrange(widthB):
			colors = B_prime_YIQ[i,j]
			RGB = colorsys.yiq_to_rgb(colors[0],colors[1],colors[2])
			B_prime_RGB[i,j] = RGB

	stop_time = time.time()
	print "conversion to RGB complete"
	print "Serial: %f" % (stop_time - start_time)

	borderless = B_prime_RGB[3:heightB-2,3:widthB-2]
	borderlessoriginal = imageB[3:heightB-2,3:widthB-2]

	result = plt.imshow(borderless)
	plt.show()

	scipy.misc.imsave(B_prime_name,borderless)
	# scipy.misc.imsave(B_name_1,borderlessoriginal)


