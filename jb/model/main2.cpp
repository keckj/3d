#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "CL/cl.h"
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


int runCL(unsigned char *image, int img_width, int img_height, const float *filter, int filter_size)
{
	cl_program program[0];
	cl_kernel kernel[1];

	cl_command_queue cmd_queue;
	cl_context context;

	cl_platform_id platform = NULL;
	cl_device_id cpu = NULL, device = NULL;

	cl_int err = 0;



	// List all available devices on every available platoforms
	//show_devices();

	// Find the default platform
	checkOpenCLError(clGetPlatformIDs(1, &platform, NULL),  __LINE__);

	// Find the CPU CL device, as a fallback
	checkOpenCLError(clGetDeviceIDs(platform , CL_DEVICE_TYPE_CPU, 1, &cpu, NULL), __LINE__);

	// Find the GPU CL device, this is what we really want
	// If there is no GPU device is CL capable, fall back to CPU

	checkOpenCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL), __LINE__);
	
	if (err != CL_SUCCESS) {
		device = cpu;
		printf("\nCPU has been chosen\n");
	}
	else {
		printf("\nGPU has been chosen\n");
	}




	// Now create a context to perform our calculation with the 
	// specified device 
	context = clCreateContext(0, 1, &device, NULL, NULL, &err);
	checkOpenCLError(err, __LINE__);	

	// And also a command queue for the context
	cmd_queue = clCreateCommandQueue(context, device, 0, NULL);


	printf("\nBuilding programm...\n");

	// Load the program source from disk
	// The kernel/program is the project directory
	const char * filename = "example.cl";
	char *program_source = load_program_source(filename);
	program[0] = clCreateProgramWithSource(context, 1, (const char**)&program_source,
			NULL, &err);
	checkOpenCLError(err, __LINE__);	

	const char build_options[] = "-w";
	err = clBuildProgram(program[0], 1, &device, build_options, NULL, NULL);

	if (err != CL_SUCCESS) {
		cl_build_status status;
		size_t logSize;
		char *programLog;

		// check build error and build status first
		clGetProgramBuildInfo(program[0], device, CL_PROGRAM_BUILD_STATUS,
				sizeof(cl_build_status), &status, NULL);

		// check build log
		clGetProgramBuildInfo(program[0], device,
				CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

		programLog = (char*) calloc (logSize+1, sizeof(char));

		clGetProgramBuildInfo(program[0], device,
				CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);

		printf("Build failed; error=%d, status=%d, programLog:\n\n%s", err, status, programLog);

		free(programLog);

	}

	printf("\nProgramm builded !\nTransferring data...");


	// Now create the kernel "objects" that we want to use in the example file 
	kernel[0] = clCreateKernel(program[0], "add", &err);
	checkOpenCLError(err, __LINE__);	

	cl_mem d_input_image, d_output_image, d_filter;

	// Input image 
	const size_t img_buffer_size = img_width * img_height * sizeof(char);
	d_input_image = clCreateBuffer(context, CL_MEM_READ_ONLY, img_buffer_size, NULL, NULL);
	checkOpenCLError(clEnqueueWriteBuffer(cmd_queue, d_input_image, CL_TRUE, 0, img_buffer_size, (void*) image, 0, NULL, NULL), __LINE__);

	// Input filter
	const size_t filter_buffer_size = filter_size * sizeof(float);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_buffer_size , NULL, NULL);
	checkOpenCLError(clEnqueueWriteBuffer(cmd_queue, d_filter, CL_TRUE, 0, filter_buffer_size, (void*) filter, 0, NULL, NULL), __LINE__);

	// Output image 
	d_output_image = clCreateBuffer(context, CL_MEM_READ_WRITE, img_buffer_size, NULL, NULL);

	// Get all of the stuff written and allocated 
	clFinish(cmd_queue);

	printf("\nData on device !\n");

	// Now setup the arguments to our kernel
	checkOpenCLError(clSetKernelArg(kernel[0],  0, sizeof(cl_mem), &d_input_image), __LINE__);
	checkOpenCLError(clSetKernelArg(kernel[0],  1, sizeof(cl_mem), &d_output_image), __LINE__);
	checkOpenCLError(clSetKernelArg(kernel[0],  2, sizeof(cl_mem), &d_filter), __LINE__);
	checkOpenCLError(clSetKernelArg(kernel[0],  3, sizeof(int) , &img_width), __LINE__);
	checkOpenCLError(clSetKernelArg(kernel[0],  4, sizeof(int) , &img_height), __LINE__);
	checkOpenCLError(clSetKernelArg(kernel[0],  5, sizeof(int), &filter_size), __LINE__);

	// Run the calculation by enqueuing it and forcing the 
	// command queue to complete the task
	const size_t Block_Size = 1;
	const size_t global_work_size[2] = { ((img_width - 1)/ (Block_Size * Block_Size) + 1) * Block_Size,
		((img_height - 1)/ (Block_Size * Block_Size) + 1) * Block_Size };
	const size_t local_work_size[2] = { Block_Size, Block_Size }; 

	printf("Gridsize : (%i, %i, %i)\nBlocksize : (%i, %i , %i)\n", (int) global_work_size[0], (int) global_work_size[1], 1, (int) local_work_size[0], (int) local_work_size[1], 1);

	checkOpenCLError(clEnqueueNDRangeKernel(cmd_queue, kernel[0], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL), __LINE__);

	clFinish(cmd_queue);

	// Once finished read back the results from the answer 
	// array into the results array
	checkOpenCLError(clEnqueueReadBuffer(cmd_queue, d_output_image, CL_TRUE, 0, img_buffer_size, image, 0, NULL, NULL), __LINE__);
	clFinish(cmd_queue);

	clReleaseMemObject(d_input_image);
	clReleaseMemObject(d_output_image);
	clReleaseMemObject(d_filter);

	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);

	return CL_SUCCESS;
}




int main (void) {

	// Load image into memory 
	unsigned int img_load_error;
	unsigned char *image;
	unsigned int img_width, img_height;


	const float filter[] = {0.09474f, 0.11832f, 0.09474f, 0.11838f, 0.14776f, 0.11831f, 0.09474f, 0.11832f, 0.09474f};
	size_t filter_size = 3;

	using namespace cv;
	Mat test_image;
	test_image = imread("1.jpg", CV_LOAD_IMAGE_COLOR);


	printf("Kernel launch !");

	// Do the OpenCL calculation
	//runCL(image, img_width, img_height, filter, filter_size);

	printf("Kernel finished !");
	/*
	for (unsigned int i = 0; i < img_width; i++) {
		for (unsigned int j = 0; j < img_height; j++) {
			int pos = i + j*img_width;
			printf("\nPosition (%i,%i) => Pixel(%i,%i,%i,%i)", i,j, image[pos], image[pos + 1], image[pos + 2], image[pos + 3]);	
		}
	}
	img_load_error = lodepng_encode32_file("out.png", image, img_width, img_height);
	if(img_load_error) printf("error %u: %s\n", img_load_error, lodepng_error_text(img_load_error));

*/

	// Free up memory
	free(image);

	return 0;
}




