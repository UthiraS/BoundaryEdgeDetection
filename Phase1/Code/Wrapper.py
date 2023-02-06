#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:
# from PIL import Image
import math

import matplotlib.pyplot as plot
import pandas as pd
from sklearn.cluster import KMeans
import os
import sys
import numpy as np
import cv2
import imutils





"""
Load Images
"""
def load_imgs(folder_name):
	#Storing all the images in the variable imgs
	imgs=[]
	for filename in os.listdir(folder_name):
		img = cv2.imread(os.path.join(folder_name,filename))
		if img is not None:
			imgs.append(img)
		else:
			print("No image found in this folder!!")
	return imgs

"""
Show Images
"""
def show_img(imgs):

	for img in imgs:
		cv2.namedWindow("Show Image")
		cv2.imshow("Input", img)


"""
Print Results of a filter Bank in Matplot
"""
def print_filterbank_results_matplot(Filter,file_name, cols):
	i =0
			
	rows = math.ceil(len(Filter )/cols)	
	plot.subplots(rows, cols, figsize=(15,15))
	for index in range(len(Filter )):
		plot.subplot(rows, cols, index+1)
		plot.axis('off')
		plot.imshow(Filter [index], cmap='gray')
	
	plot.savefig(file_name)
	plot.close()

"""
Chi Square distance
"""
def chisquareDistance(input, bins, filter_bank):

	chi_sq_dis = []
	N = len(filter_bank)
	i = 0
	while i < N:
		left_mask = filter_bank[i]
		right_mask = filter_bank[i+1]		
		tmp = np.zeros(input.shape)
		csd = np.zeros(input.shape)
		min_bin = np.min(input)
	

		for bin in range(bins):
			tmp[input == bin+min_bin] = 1
			g_i = cv2.filter2D(tmp,-1,left_mask)
			h_i = cv2.filter2D(tmp,-1,right_mask)
			term1 = (g_i - h_i)**2
			term2 = 1/(g_i + h_i + np.exp(-7))
			csd += term1*term2

		csd /= 2
		chi_sq_dis.append(csd)
		i = i+2
    	

	return chi_sq_dis


"""
Generate Gaussian Filter:	
"""
def gaussian(sigma,kernel_size):
	sigma_x, sigma_y = sigma
	Gauss = np.zeros([kernel_size, kernel_size])
	# x = np.linspace(0,kernel_size)
	# y = np.linspace(0,kernel_size)
	if(kernel_size/2):
		index = kernel_size/2
	else:
		index = (kernel_size - 1)/2
	x,y= np.meshgrid(np.linspace(-index,index,kernel_size),np.linspace(-index,index,kernel_size))
	term1 = 0.5/(np.pi*sigma_x*sigma_y)
	
	term2 = np.exp(-((np.square(x)/(np.square(sigma_x))+(np.square(y)/(np.square(sigma_y))))))/2
	Gauss = term1*term2
	return Gauss
"""
Generate Difference of Gaussian Filter Bank: (DoG)
Display all the filters in this filter bank and save image as DoG.png,
use command "cv2.imwrite(...)"
"""
def DOG_FilterBank(orientation_values,scale_values,kernel_size):

	DOG= []
	"""
	Sobel Filter - Fixed 
	"""

	Sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	Sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])


	for scale in scale_values:
		sigma = [scale, scale]
		Gauss = gaussian(sigma,kernel_size)

		DOG_X = cv2.filter2D(Gauss,-1,Sobel_x)
		DOG_Y = cv2.filter2D(Gauss,-1,Sobel_y)
		for ori in range(orientation_values):
			curr_orientation = ori *2 *np.pi /orientation_values
			DOG_Filter = (DOG_X * np.cos(curr_orientation)) +(DOG_Y *np.sin(curr_orientation))
			DOG.append(DOG_Filter)
	# np_array = np.array(DOG, dtype=np.int32)
	# DOG= np_array.astype(np.float32)
	# DOG.astype('float32') 
	# print (DOG)
	fig, axs = plot.subplots(len(scale_values),orientation_values,figsize=(orientation_values,len(scale_values)))
	for i in range(len(scale_values)):
		for j in range(orientation_values):
				
				axs[i, j].imshow(DOG[i*orientation_values+j],cmap='gray')
				axs[i, j].axis('off')

	# plot.show()
	# plot.savefig("/home/uthira/usivaraman_hw0/Phase1/Results/DOG.png")
	plot.close()
	return DOG




"""
Generate Leung-Malik Filter Bank: (LM)
Display all the filters in this filter bank and save image as LM.png,
use command "cv2.imwrite(...)"
"""
def LM_FilterBank(orientation_values,scale_values,kernel_size):

	LM =[]
	
	"""
	Defining Scales :
	Derivate filter : 3
	Laplacian of Gaussian filter : 8 (sigma+sigma*3)
	Gaussian : 4
	"""
	Derivatives_scale = scale_values[0:3]
	GaussinaFilter_scale = scale_values
	LOGFilter_scale =[]
	for i in range(len(scale_values)):
		scale = scale_values[i]
		LOG_scale_value = scale + 3 *scale
		LOGFilter_scale.append(LOG_scale_value)

	"""
		First and Second Derivatives of Gauss Filter
	"""
	FirstD =[]
	SecondD=[]
	for scale in Derivatives_scale :
		sigma = [scale, scale]

		
		del_x = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
		del_y = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]])
		Gauss = gaussian(sigma,kernel_size)		
		FirstD_x = cv2.filter2D(Gauss,-1,del_x)
		FirstD_y = cv2.filter2D(Gauss,-1,del_y)
		SecondD_x = cv2.filter2D(FirstD_x,-1,del_x)
		SecondD_Y = cv2.filter2D(FirstD_y,-1,del_y)
		for ori in range(orientation_values):
			curr_orientation = ori *2 *np.pi /orientation_values
			FirstD_Filter = (FirstD_x* np.cos(curr_orientation)) +(FirstD_y *np.sin(curr_orientation))
			SecondtD_Filter = (SecondD_x* np.cos(curr_orientation)) +(SecondD_Y *np.sin(curr_orientation))
			FirstD.append(FirstD_Filter)
			SecondD.append(SecondtD_Filter)

	"""
	Laplacian of Gaussian Filter
	
	"""
	LOG=[]
	if(kernel_size/2):
			index = kernel_size/2
	else:
			index = (kernel_size - 1)/2
	for scale in LOGFilter_scale:
		sigma = [scale, scale]
		Gauss = gaussian(sigma, kernel_size)
		Log_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
		LOG_Filter= cv2.filter2D(Gauss,-1,Log_kernel)
		LOG.append(LOG_Filter)
		# x,y= np.meshgrid(np.linspace(-index,index,kernel_size),np.linspace(-index,index,kernel_size))
		# term1 = -1/(np.pi*np.square(sigma[0])*np.square(sigma[1]))
		# term2 = (1-((np.square(x)/(np.square(sigma[0]))+(np.square(y)/(np.square(sigma[1]))))))
		# term3 = np.exp(-((np.square(x)/(np.square(sigma[0]))+(np.square(y)/(np.square(sigma[1]))))))/2
		# LOG_filter = term1*term2*term3
		# LOG.append(LOG_filter)

	"""
	Gaussian Filter
	"""
	Gaussian=[]
	for scale in GaussinaFilter_scale:
		sigma = [scale, scale]
		Gaussian.append(gaussian(sigma, kernel_size))
	
	LM = FirstD +SecondD + LOG + Gaussian

	fig, axs = plot.subplots(len(scale_values),orientation_values,figsize=(orientation_values,len(scale_values)))
	for i in range(len(scale_values)):
		for j in range(orientation_values):
				
				axs[i, j].imshow(LM[i*orientation_values+j],cmap='gray')
				axs[i, j].axis('off')
	# plot.show()
	return LM
"""
Generate Sine Wave:	
"""
def sinewave(frequency,kernel_size,angle):
	
	
	index = (kernel_size - 1)/2

	x, y = np.meshgrid(np.linspace(-index, index+1, kernel_size), np.linspace(-index, index+1, kernel_size))
	value= x * np.cos(angle) + y * np.sin(angle)
	sin2d = np.sin(value* 2 * np.pi * frequency/kernel_size)

	return sin2d



"""
Generate Gabor Filter Bank: (Gabor)
Display all the filters in this filter bank and save image as Gabor.png,
use command "cv2.imwrite(...)"
"""
def Gabor_FilterBank(orientation_values,scale_values,frequency_values,kernel_size):

	Gabor =[]
	for scale in scale_values:
		sigma = [scale, scale]
		Gauss = gaussian(sigma,kernel_size)
		for angle in range(orientation_values):
			for frequency in frequency_values:
				SinewaveFilter =sinewave(frequency,kernel_size,angle)
				Gabor_Filter = Gauss*SinewaveFilter
				Gabor.append(Gabor_Filter)
	fig, axs = plot.subplots(len(scale_values),orientation_values,figsize=(orientation_values,len(scale_values)))
	for i in range(len(scale_values)):
		for j in range(orientation_values):
				
				axs[i, j].imshow(Gabor[i*orientation_values+j],cmap='gray')
				axs[i, j].axis('off')
	# plot.show()
	# plot.savefig("/home/uthira/usivaraman_hw0/Phase1/Results/Gabor.png")
	plot.close()
	return Gabor




"""
Generate Half-disk masks
Display all the Half-disk masks and save image as HDMasks.png,
use command "cv2.imwrite(...)"
"""
def HalfDisk(radius, angle):
	# radius = 3
	# Half_disk_masks =np.ones((8,8))
	# y,x = np.ogrid[-3: 3+1, 0: 3+1]
	# Half_disk_masks = x**2+y**2 <= 3**2
	# Half_disk_masks = np.array(Half_disk_masks)
	# print(Half_disk_masks)
	# print_filterbank_results_matplot(Half_disk_masks,'/home/uthira/CV/Cvis_HW0/Phase1/Results/Filter/HalfDiskMasks.png', 8)
	size = 2*radius + 1
	centre = radius
	half_disk = np.zeros([size, size])
	for i in range(radius):
		for j in range(size):
			distance = np.square(i-centre) + np.square(j-centre)
			if distance <= np.square(radius):
				half_disk[i,j] = 1
    
	
	half_disk = imutils.rotate(half_disk, angle)
	half_disk[half_disk<=0.5] = 0
	half_disk[half_disk>0.5] = 1
	return half_disk
def halfdiskFilters(radii, orientations):
	filter_bank = []
	for radius in radii:
		filter_bank_pairs = []
		temp = []
		for orientation in range(orientations):
			angle = orientation * 360 / orientations
			half_disk_filter = HalfDisk(radius, angle)
			temp.append(half_disk_filter)

        #to make pairs
		i = 0
		while i < orientations/2:
			filter_bank_pairs.append(temp[i])
			filter_bank_pairs.append(temp[i+int((orientations)/2)])
			i = i+1

		filter_bank+=filter_bank_pairs
	
	
	return filter_bank	

def pblite_edges(T_g, B_g, C_g, Canny_edge, Sobel_edges, weights):
	Canny_edge = cv2.cvtColor(Canny_edge, cv2.COLOR_BGR2GRAY)
	Sobel_edges = cv2.cvtColor(Sobel_edges, cv2.COLOR_BGR2GRAY)
	T1 = (T_g + B_g + C_g)/3
	w1 = weights[0]
	w2 = weights[1]
	T2 = (w1 * Canny_edge) + (w2 * Sobel_edges)
	# print(T1.shape)
	# print(T2.shape)
	pb_lite_output = np.multiply(T1, T2)
	return pb_lite_output


"""
Main Function
"""
def main():
		
		"""Derivative of Gaussian Filter"""	
		DOG_FB = DOG_FilterBank(18,[3,3],36)		
		# print_filterbank_results_matplot(DOG_FB,'/home/uthira/CV/Cvis_HW0/Phase1/Results/Filter/DOG.png',8)
	
		

		"""LM Filter"""	
		LM_FB_small = LM_FilterBank(6,[1,np.sqrt(2),2,2*np.sqrt(2)],49)		
		LM_FB_large= LM_FilterBank(6,[np.sqrt(2),2,2*np.sqrt(2),4],49)	
		# print_filterbank_results_matplot(LM_FB_small,'/home/uthira/CV/Cvis_HW0/Phase1/Results/Filter/LM_small.png',12)
		# print_filterbank_results_matplot(LM_FB_large,'/home/uthira/CV/Cvis_HW0/Phase1/Results/Filter/LM_large.png',12)
		LM_FB = LM_FB_small + LM_FB_large
		# print_filterbank_results_matplot(LM_FB,'/home/uthira/CV/Cvis_HW0/Phase1/Results/Filter/LM.png',12)

		fig, axs = plot.subplots(8,6,figsize=(6,8))
		for i in range(8):
			for j in range(6):
					
					axs[i, j].imshow(LM_FB[i*6+j],cmap='gray')
					axs[i, j].axis('off')
		# plot.show()
		

		"""Gabor Filter"""	
		Gabor_FB = Gabor_FilterBank(6,[20,45],[3,4,6],49)		
		# print_filterbank_results_matplot(Gabor_FB,'/home/uthira/CV/Cvis_HW0/Phase1/Results/Filter/Gabor.png',8)
	
		
		"""
		Total Filter Bank 
		"""
		

		textron_gradients = []
		brightness_gradients = []
		color_gradients = []
		half_disk_filter_bank = halfdiskFilters([2,5,10,20,30], 16)
		index =0
		
		images_folder = "/home/uthira/usivaraman_hw0/Phase1/Code/BSDS500/Images/"
		results_folder = "/home/uthira/usivaraman_hw0/Phase1/Code/Results/"
		
		image_files = os.listdir(images_folder)
		for img_name in image_files:
			print("image name",img_name)
			img_path = images_folder + img_name
			img = cv2.imread(img_path)
			# plot.imshow(img, cmap = "gray")
			# plot.show()
			print("index:",index)
			image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			filterbank = DOG_FB + LM_FB + Gabor_FB
			filtered_images =[]
			for fb in filterbank:
				Results = cv2.filter2D(image_gray,-1, fb)	
				filtered_images.append(Results)
			filename = results_folder+"Filter/"+"Fbank_"+img_name
			# cv2.imwrite(os.path.join(path , image_name), img)
		# cv2.waitKey(0)
			# print_filterbank_results_matplot(filtered_images,filename,8)		
			# cv2.imshow("output", Results)
			# cv2.waitKey()
			
			"""
			Generate Texton Map
			Filter image using oriented gaussian filter bank
			"""
			filtered_images= np.array(filtered_images)
			# print("Shape of Filtered Images")
			# print(filtered_images.shape)
			# print("Length of Filterbank")
			# print(len(filterbank))
			f,x,y = filtered_images.shape
			input_mat = filtered_images.reshape([f, x*y])
			input_mat = input_mat.transpose()
			# print("Input to Kmeans clustering")
			# print(input_mat)
			# print(input_mat.shape)
			

		

			"""
			Generate texture ID's using K-means clustering
			Display texton map and save image as TextonMap_ImageName.png,
			use command "cv2.imwrite('...)"
			"""
			print("Textron Maps")
			kmeans = KMeans(n_clusters=64, init='k-means++', max_iter=300, n_init=2, random_state=0)
			kmeans.fit(input_mat)
			labels = kmeans.predict(input_mat)
			texton_image = labels.reshape([x,y])
			print(texton_image.shape)
			image_name ="Texton_"+img_name
			# cv2.imwrite(os.path.join(results_folder , image_name), T_g)
			plot.imsave(os.path.join(results_folder , image_name),  texton_image, cmap = None)
			# textron_maps.append(texton_image)
			"""
			Generate Texton Gradient (Tg)
			Perform Chi-square calculation on Texton Map
			Display Tg and save image as Tg_ImageName.png,
			use command "cv2.imwrite(...)"
			"""
			T_g = chisquareDistance(texton_image, 64, half_disk_filter_bank)
			T_g = np.array(T_g)
			T_g = np.mean(T_g, axis = 0)
			#plt.imshow(T_g)
			#plt.show()	
			textron_gradients.append(T_g)
			# plot.imshow(T_g, cmap=None)
			# plot.show()	
			# plot.close()
			# T_g = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
			image_name ="Tg_"+img_name
			# cv2.imwrite(os.path.join(results_folder , image_name), T_g)
			plot.imsave(os.path.join(results_folder , image_name),  T_g, cmap = None)

			# plot.imshow(texton_image)	
			# plot.show()	
			# plot.close()
			# plot.imsave(os.path.join("/home/uthira/CV/Cvis_HW0/Phase1/Results/Textron_map/", "TextonMap_"+str(index)) , texton_image)
			

			"""
			Generate Brightness Map
			Perform brightness binning 
			"""
			print("generating brightness maps..")
			
			# for i,image in enumerate(images):
			image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			x,y = image_gray.shape
			input_mat = image_gray.reshape([x*y,1])
			kmeans = KMeans(n_clusters = 16, n_init = 4)
			kmeans.fit(input_mat)
			labels = kmeans.predict(input_mat)
			brightness_image = labels.reshape([x,y])
			print(brightness_image.shape)
			image_name ="Bright_"+img_name
			# cv2.imwrite(os.path.join(results_folder , image_name), T_g)
			plot.imsave(os.path.join(results_folder , image_name),  brightness_image, cmap = None)
			"""
			Generate Brightness Gradient (Bg)
			Perform Chi-square calculation on Brightness Map
			Display Bg and save image as Bg_ImageName.png,
			use command "cv2.imwrite(...)"
			"""
			# brightness_maps.append(brightness_image)
			B_g = chisquareDistance(brightness_image, 16, half_disk_filter_bank)
			B_g = np.array(B_g)
			B_g = np.mean(B_g, axis = 0)
			#plt.imshow(B_g)
			#plt.show()
			brightness_gradients.append(B_g)
			image_name ="Bg_"+img_name
			# cv2.imwrite(os.path.join(results_folder , image_name), B_g)
			plot.imsave(os.path.join(results_folder , image_name),  B_g, cmap = None)
				# plot.imshow(brightness_image)
				# plot.show()
				# plt.imsave(folder_name + "results/Brightness_map/BrightnessMap_" + file_names[i], brightness_image)

			

			"""
			Generate Color Map
			Perform color binning or clustering
			"""
			print("generating color maps..")
			
			
			x,y,c = img.shape
			input_mat = img.reshape([x*y,c])
		
			kmeans = KMeans(n_clusters =16, n_init = 4)
			kmeans.fit(input_mat)
			labels = kmeans.predict(input_mat)
			color_image = labels.reshape([x,y])
			image_name ="Color_"+img_name
			# cv2.imwrite(os.path.join(results_folder , image_name), T_g)
			plot.imsave(os.path.join(results_folder , image_name),  color_image, cmap = None)
			# color_maps.append(color_image)
			"""
			Generate Color Gradient (Cg)
			Perform Chi-square calculation on Color Map
			Display Cg and save image as Cg_ImageName.png,
			use command "cv2.imwrite(...)"
			"""
			C_g = chisquareDistance(color_image,16, half_disk_filter_bank)
			C_g = np.array(C_g)
			C_g = np.mean(C_g, axis = 0)
			# print(color_image.shape)
			#plt.imshow(C_g)
			#plt.show()
			color_gradients.append(C_g)
			image_name ="Cg_"+img_name
			# cv2.imwrite(os.path.join(results_folder , image_name), C_g)
			plot.imsave(os.path.join(results_folder , image_name),  C_g, cmap = None)
				#plt.imshow(color_image)
				#plt.show()			
				# plt.imsave(folder_name + "results/Color_map/ColorMap_"+ file_names[i], color_image)
			"""
			Combine responses to get pb-lite output
			Display PbLite and save image as PbLite_ImageName.png
			use command "cv2.imwrite(...)"
			"""
			print("generating pb lite output..")
			# for i in range(len(images)):	
			sobel_pb = cv2.imread("/home/uthira/usivaraman_hw0/Phase1/Code/BSDS500/SobelBaseline/" + img_name)
			canny_pb = cv2.imread("/home/uthira/usivaraman_hw0/Phase1/Code/BSDS500/CannyBaseline/" + img_name)
			pb_edge = pblite_edges(T_g, B_g, C_g, canny_pb, sobel_pb, [0.5,0.5])
			print("Final Output")
			# plot.imshow(pb_edge, cmap = "gray")
			plot.show()
			# pb_edge =cv2.cvtColor(pb_edge, cv2.COLOR_BGR2GRAY)
			image_name ="PbLite_"+img_name
			# cv2.imwrite(os.path.join(results_folder , image_name), pb_edge)
			plot.imsave(os.path.join(results_folder , image_name), pb_edge, cmap = "gray")
			# /home/uthira/CV/Cvis_HW0/Phase1/Results/Pblite/Figure_1.png
			
			index = index +1
    
if __name__ == '__main__':
    main()
 



