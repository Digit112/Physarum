#include <stdio.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

// Quick library used to get setup and error handling code out of main.cpp
// It creates an object which creates and holds the OpenCL platform, device, context, queue, program, and kernels.
class compute {
public:
	cl_platform_id p;
	cl_device_id d;
	cl_context c;
	cl_command_queue q;
	cl_program prog;

	cl_kernel k_hullo;
	cl_kernel k_blur;

	compute() {
		cl_int err;

		// Get Platform and Device.
		err = clGetPlatformIDs(1, &p, NULL);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		// Create Context
		c = clCreateContext(NULL, 1, &d, NULL, NULL, &err);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		// Create Command Queue
		q = clCreateCommandQueueWithProperties(c, d, NULL, &err);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		// Get info
		char buff[256];
		err = clGetDeviceInfo(d, CL_DEVICE_NAME, 512, buff, NULL);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
		printf("Running on %s.\n", buff);
	}

	void build_program(const char* fn) {
		// Build cl code
		cl_int err;

		char* source = (char*) malloc(8192);
		FILE* fin = fopen(fn, "r");
		size_t source_n = fread(source, 1, 8191, fin);
		fclose(fin);
		printf("Read %ld bytes of source.\n", source_n);
		source[source_n] = '\0';

		prog = clCreateProgramWithSource(c, 1, (const char**) &source, &source_n, &err);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		err = clBuildProgram(prog, 1, &d, NULL, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error on %d: %d\n", __LINE__-1, err);
			
			// On build failure, retrieve and display the build log.
			size_t log_n = 0;
			err = clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_n);
			if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

			char log[log_n+1];
			log[log_n] = '\0';
	
			err = clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, log_n, log, NULL);
			if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

			printf("%s\n", log);	
		}

		free(source);
	}

	void get_physarum_kernels() {
		cl_int err;

		k_hullo = clCreateKernel(prog, "hullo", &err);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		k_blur = clCreateKernel(prog, "blur", &err);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
	}
};

