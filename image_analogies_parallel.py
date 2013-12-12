from sys import argv
import numpy as np
import time
import colorsys

import matplotlib.image as img
import matplotlib.pyplot as plt
import scipy.misc as scipy
import cv2
# from cv2 import cv

from Cheetah.Template import Template
# from pycuda.elementwise import ElementwiseKernel

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit

from mpi4py import MPI
import math

# Define the CUDA saxpy kernel as a string.
analogy_kernel_source = \
"""
#echo $HEADER #

__device__ int get_global(int i, int j, int cols){
  return i * cols + j;  
}

#for $i in range(3)
__global__ void #echo $YIQ[$i] #_kernel(double* image, double* converted, int total_pixels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_pixels){
        converted[gid] = image[3 * gid] * $TO_YIQ[$i][0] + image[3 * gid + 1] * $TO_YIQ[$i][1] + image[3 * gid + 2] * $TO_YIQ[$i][2];
    }
}
#end for

__global__ void rgb_kernel(double* yiq, double* rgb, int total_pixels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_pixels){
        double pix[3];
        #for $i in range(3)
            pix[$i] = yiq[3 * gid + $i];
        #end for
        rgb[3 * gid] = fmax(0., (pix[0] + $TO_RGB[0][0] * pix[1] + $TO_RGB[0][1] * pix[2]) / 255.);
        rgb[3 * gid + 1] = fmax(0., (pix[0] + $TO_RGB[1][0] * pix[1] + $TO_RGB[1][1] * pix[2]) / 255.);
        rgb[3 * gid + 2] = fmax(0., (pix[0] + $TO_RGB[2][0] * pix[1] + $TO_RGB[2][1] * pix[2]) / 255.);
    }
}

__global__ void feature_kernel(double* image, double* feature_vecs, int rows, int cols){
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = get_global(xid, yid, cols);
    int mini_id = 0;
    if (xid < rows - 2 && xid > 1 && yid < cols - 2 && yid > 1){
        #set $index = 0
        #for $i in range(-2, 3, 1)
            #for $j in range(-2, 3, 1)
                mini_id = get_global(xid + $i, yid + $j, cols);
                feature_vecs[25 * gid + $index] = image[mini_id] * $GAUSSIAN_KERNEL[$index];
                #set $index += 1
            #end for
        #end for
    }
}

__global__ void similar_kernel(double* featuresA, double* featuresB, double* y_imageA_prime, double* y_imageB_prime, int total_pixels_B){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_pixels_B){
        
        double min_d = 1.0 / 0.0;
        double distance = 0;
        int match = 0;
        double to_square = 0;

        for (int i = 0; i < $TOTAL_FEATURES_A; i++) {
            if (i % 25 == 24) {
                if (distance < min_d){
                    min_d = distance;
                    match = i / 25;
                }
                distance = 0;
            }
            to_square = featuresA[i] - featuresB[gid * 25 + i % 25];
            distance += to_square * to_square;
        }
        if (match != 0){
            y_imageB_prime[gid] = y_imageA_prime[match];
        }
    }   
}

__global__ void ann_helper_kernel(double* y_imageA_prime, double* y_imageB_prime, int* match_list, int* j_minlist, int* i_minlist){
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = get_global(xid, yid, $COLS_B);
    
    if (xid < #echo $ROWS_B - 2# && xid > 1 && yid < #echo $COLS_B - 2# && yid > 1){
        int imin = i_minlist[gid];
        int jmin = j_minlist[gid];
        int loc = get_global(imin, jmin, $COLS_A);
        match_list[gid] = loc;
        y_imageB_prime[gid] = y_imageA_prime[loc];
    }   
}
"""

def MPI_coherence(comm, rank, feature_vectorA, feature_vectorB, imageB_match, y_imageA_prime, y_imageB_prime):
    '''
    Calculate coherence in parallel
    '''
    feature_vectorA = comm.bcast(feature_vectorA, root = 0)
    feature_vectorB = comm.bcast(feature_vectorB, root = 0)
    y_imageA_prime = comm.bcast(y_imageA_prime, root = 0)
    imageB_match = comm.scatter(imageB_match, root = 0)
    y_imageB_prime = comm.scatter(y_imageB_prime, root = 0)
    gaussian_kernel = np.float64(np.array([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.)

    heightB, widthB, _ = np.int32(feature_vectorB.shape)
    heightA, widthA, _ = np.int32(feature_vectorA.shape)

    segHeight, segWidth = np.int32(imageB_match.shape)

    for i in range(2, segHeight - 1, 1):
        for j in range(2, segWidth - 1, 1): 
            i_actual = i + rank * segHeight      
            match = (i,j)
            coh_distance = float("inf")
            for i_plus in range(-2,1,1):
                for j_plus in range(-2, 3, 1):
                    i_new, j_new = (imageB_match[(i + i_plus) % segHeight][(j + j_plus) % segWidth] / widthA, imageB_match[(i + i_plus) % segHeight][(j + j_plus) % segWidth] % widthA)
                    d = np.linalg.norm(gaussian_kernel * (feature_vectorB[i_actual % heightB][j] - feature_vectorA[i_new][j_new]))
                    if i_plus == 0 and j_plus == 0:
                        ann_distance = d
                        break                    
                    if d < coh_distance:
                        coh_distance = d
                        match = (i_new, j_new)
            # if feature_vectorB[i][j][12] < 0.2:
            #     y_imageB_prime[i][j] = 0
            # el
            if coh_distance < 3 * ann_distance:
                y_imageB_prime[i][j] = y_imageA_prime[match[0]][match[1]]
                imageB_match[i][j] = match[0] * widthA + match[1]

    y_imageB_prime = comm.gather(y_imageB_prime, root = 0)
    return y_imageB_prime

def get_analogy_template(heightA, widthA, heightB, widthB, num_pixelsA):    
    # Constants for Cheetah
    analogy_template = Template(analogy_kernel_source)
    analogy_template.TO_YIQ = [[0.299, 0.587, 0.114], [0.595716, -0.274453, -0.321263], [0.211456, -0.522591, 0.311135]]
    analogy_template.TO_RGB = [[0.9563, 0.6210], [-0.2721, -0.6474], [-1.107, 1.7046]]
    analogy_template.GAUSSIAN_KERNEL = np.float64(np.array([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.)
    analogy_template.YIQ = ["y", "i", "q"]
    analogy_template.TOTAL_FEATURES_A = np.float32(num_pixelsA * 25)
    analogy_template.COLS_A = np.float32(widthA)
    analogy_template.COLS_B = np.float32(widthB)
    analogy_template.ROWS_B = np.float32(heightB)
    analogy_template.CONSTANT = 0.1
    analogy_template.HEADER = "#include <math.h>"
    return analogy_template

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        if len(argv) != 5:
            print "Usage: python", argv[0], "[imageA] [imageA'] [imageB] [output_file]"
            exit()

        # Read image files
        imageA = img.imread(argv[1])
        imageA_prime = img.imread(argv[2])
        imageB = img.imread(argv[3])
        B_prime_name = argv[4]

        # determine the shape of each image
        heightA, widthA, _ = imageA.shape
        heightB, widthB, _ = imageB.shape
        num_pixelsA = np.int32(heightA * widthA)
        num_pixelsB = np.int32(heightB * widthB)

        print "Processing A %d x %d image" % (heightA,widthA)
        print "Processing B %d x %d image" % (heightB,widthB)

        # Kernel execution block and grid sizes
        block_size_1D = (32, 1, 1)
        grid_size_A_1D = (int(np.ceil(float(num_pixelsA) / block_size_1D[0])), 1, 1)
        grid_size_B_1D = (int(np.ceil(float(num_pixelsB) / block_size_1D[0])), 1, 1)

        block_size_2D = (32, 32, 1)
        grid_size_A_2D = (int(np.ceil(float(heightA) / block_size_2D[0])), int(np.ceil(float(widthA) / block_size_2D[0])), 1)
        grid_size_B_2D = (int(np.ceil(float(heightB) / block_size_2D[0])), int(np.ceil(float(widthB) / block_size_2D[0])), 1)

        # Get kernel functions
        analogy_module = nvcc.SourceModule(get_analogy_template(heightA, widthA, heightB, widthB, num_pixelsA))
        y_kernel = analogy_module.get_function("y_kernel")
        i_kernel = analogy_module.get_function("i_kernel")
        q_kernel = analogy_module.get_function("q_kernel")
        rgb_kernel = analogy_module.get_function("rgb_kernel")
        feature_kernel = analogy_module.get_function("feature_kernel")
        similar_kernel = analogy_module.get_function("similar_kernel")
        ann_helper_kernel = analogy_module.get_function("ann_helper_kernel")

        # Initialize the GPU event trackers for timing
        start_gpu_time = cu.Event()
        end_gpu_time = cu.Event()

        # START TIMER
        start_gpu_time.record()

        d_imageA = gpu.to_gpu(np.float64(np.array(imageA)))
        d_imageA_prime = gpu.to_gpu(np.float64(np.array(imageA_prime)))
        d_imageB = gpu.to_gpu(np.float64(np.array(imageB)))
      
        d_y_imageA = gpu.zeros((heightA, widthA), dtype = np.float64)
        y_kernel(d_imageA, d_y_imageA, num_pixelsA, block = block_size_1D, grid = grid_size_A_1D)
        y_imageA = d_y_imageA.get()

        d_y_imageA_prime = gpu.zeros((heightA, widthA), dtype = np.float64)
        y_kernel(d_imageA_prime, d_y_imageA_prime, num_pixelsA, block = block_size_1D, grid = grid_size_A_1D)
        y_imageA_prime = d_y_imageA_prime.get()

        d_y_imageB = gpu.zeros((heightB, widthB), dtype = np.float64)
        y_kernel(d_imageB, d_y_imageB, num_pixelsB, block = block_size_1D, grid = grid_size_B_1D)
        y_imageB = d_y_imageB.get()

        d_i_imageB = gpu.zeros((heightB, widthB), dtype = np.float64)
        i_kernel(d_imageB, d_i_imageB, num_pixelsB, block = block_size_1D, grid = grid_size_B_1D)
        i_imageB = d_i_imageB.get()

        d_q_imageB = gpu.zeros((heightB, widthB), dtype = np.float64)
        q_kernel(d_imageB, d_q_imageB, num_pixelsB, block = block_size_1D, grid = grid_size_B_1D)
        q_imageB = d_q_imageB.get()

        # END TIMER
        end_gpu_time.record()
        end_gpu_time.synchronize()
        gpu_time = start_gpu_time.time_till(end_gpu_time) * 1e-3

        print "Done with conversion to YIQ"
        print "GPU Time: %f" % gpu_time


        # FEATURE VECTORS

        # START TIMER
        start_gpu_time.record()

        d_feature_imageA = gpu.zeros((heightA, widthA, 25), dtype = np.float64)
        feature_kernel(d_y_imageA, d_feature_imageA, np.int32(heightA), np.int32(widthA), block = block_size_2D, grid = grid_size_A_2D)
        feature_vectorA = d_feature_imageA.get()

        d_feature_imageA_prime = gpu.zeros((heightA, widthA, 25), dtype = np.float64)
        feature_kernel(d_y_imageA_prime, d_feature_imageA_prime, np.int32(heightA), np.int32(widthA), block = block_size_2D, grid = grid_size_A_2D)
        feature_vectorA_prime = d_feature_imageA_prime.get()

        d_feature_imageB = gpu.zeros((heightB, widthB, 25), dtype = np.float64)
        feature_kernel(d_y_imageB, d_feature_imageB, np.int32(heightB), np.int32(widthB), block = block_size_2D, grid = grid_size_B_2D)
        feature_vectorB = d_feature_imageB.get()
        # similar_kernel = analogy_module.get_function("similar_kernel")

        d_y_imageB_prime = gpu.zeros((heightB, widthB), dtype = np.float64)
        d_match_list = gpu.zeros((heightB, widthB), dtype = np.int32)

        # END TIMER
        end_gpu_time.record()
        end_gpu_time.synchronize()
        gpu_time = start_gpu_time.time_till(end_gpu_time) * 1e-3

        print "feature vectors made"
        print "GPU Time: %f" % gpu_time


        # MATCHING

        # ANN search function
        def ANNalgo(flnn, featureA_prime, featureB):       
            # find the ANN!
            print "be patient, doing ANN takes a while"
            indices,dist = flnn.knnSearch(featureB, 1, params={})

            # convert the index to row and col
            jminlist = indices % widthA
            iminlist = indices / widthA
         
            d_jminlist = gpu.to_gpu(np.array(jminlist, dtype = np.int32))
            d_iminlist = gpu.to_gpu(np.array(iminlist, dtype = np.int32))

            ann_helper_kernel(d_y_imageA_prime, d_y_imageB_prime, d_match_list, d_jminlist, d_iminlist, block = block_size_2D, grid = grid_size_B_2D)
            return d_match_list.get(), d_y_imageB_prime.get()

        feature_vectorA.shape = (num_pixelsA, 25)
        feature_vectorB.shape = (num_pixelsB, 25)

        # START TIMER
        start_gpu_time.record()

        trainset = np.array(feature_vectorA, dtype=np.float32)
        params = dict(algorithm=1,trees=4)
        flnn = cv2.flann_Index(trainset,params)
        imageB_match, y_imageB_prime = ANNalgo(flnn, np.float32(feature_vectorA_prime), np.float32(feature_vectorB))

        # END TIMER
        end_gpu_time.record()
        end_gpu_time.synchronize()
        gpu_time = start_gpu_time.time_till(end_gpu_time) * 1e-3

        print "ANN completed"
        print "GPU Time: %f" % gpu_time

        # # the brute force version of nearest neighbors
        # # START TIMER
        # start_gpu_time.record()

        # d_y_imageB_prime = gpu.zeros((heightB, widthB), dtype = np.float64)
        # similar_kernel(d_feature_imageA, d_feature_imageB, d_y_imageA_prime, d_y_imageB_prime, num_pixelsB, block = block_size_1D, grid = grid_size_B_1D)
        # y_imageB_prime = d_y_imageB_prime.get()
       
        # # END TIMER
        # end_gpu_time.record()
        # end_gpu_time.synchronize()
        # gpu_time = start_gpu_time.time_till(end_gpu_time) * 1e-3

        # print "GPU Mean/Variance Time: %f" % gpu_time

        start = time.time()
        feature_vectorA.shape = (heightA, widthA, 25)
        feature_vectorB.shape = (heightB, widthB, 25)

        split_imageB_match = []
        split_y_imageB_prime = []
        for i in range(size - 1):
            split_imageB_match.append(imageB_match[i * heightB / size : (i + 1) * heightB / size])
            split_y_imageB_prime.append(y_imageB_prime[i * heightB / size : (i + 1) * heightB / size])
        split_imageB_match.append(imageB_match[(size - 1) * heightB / size:])
        split_y_imageB_prime.append(y_imageB_prime[(size - 1) * heightB / size:])
    else:
        feature_vectorA = None
        feature_vectorB = None
        y_imageA_prime = None
        split_imageB_match = None
        split_y_imageB_prime = None

    comm.barrier()
    b_prime = MPI_coherence(comm, rank, feature_vectorA, feature_vectorB, split_imageB_match, y_imageA_prime, split_y_imageB_prime)
    
    # regrouping image and converting back to RGB
    if rank == 0:
        end = time.time()
        print end - start
        y_imageB_prime = []
        for mini_list in b_prime:
            y_imageB_prime.extend(mini_list.flatten())
   
        # START TIMER
        start_gpu_time.record()
        
        yiq_imageB_prime = np.float64(np.array(zip(np.array(y_imageB_prime), i_imageB.flatten(), q_imageB.flatten())))
        d_yiq_imageB_prime = gpu.to_gpu(yiq_imageB_prime)
        d_rgb_imageB_prime = gpu.zeros((heightB, widthB, 3), dtype = np.float64)
        rgb_kernel(d_yiq_imageB_prime, d_rgb_imageB_prime, num_pixelsB, block = block_size_1D, grid = grid_size_B_1D)
        rgb_imageB_prime = np.flipud(d_rgb_imageB_prime.get())

        # END TIMER
        end_gpu_time.record()
        end_gpu_time.synchronize()
        gpu_time = start_gpu_time.time_till(end_gpu_time) * 1e-3

        print "Conversion to RGB completed"
        print "GPU Time: %f" % gpu_time

        scipy.imsave(B_prime_name, rgb_imageB_prime)