#include <math.h>
#include <stdint.h>

#include "opencl_init.hpp"

// Creates an object which represents a 2D physarum simulation. This object automatically creates a compute object, which performs all necessary OpenCL setup.
// This class keeps track of all necessary structures and implements the relevant algorithms for performing physarum simulations.
class physarum2 {
public:
	// GPU / OpenCL stuff
	compute c;

	int f;

	// Arrays have a local variant "l_", a remote (Stored on the GPU) variant "r_", and a temporary (also remote) vriant "t_"
	// The temporary variant is used to store the results of operations which cannot be performed in-place and is for internal use only.

	// Agents stored as x, y, dir float triplets.
	float* l_agents;
	cl_mem r_agents;
	cl_mem t_agents;
	uint32_t agents_n;
	
	// Deposit/chemoattractant grid
	uint16_t* l_deposit;
	cl_mem r_deposit;
	cl_mem t_deposit;

	// This matrix cn be used to quickly test if a cell is occupied. It is maintained and used only by Jones's original algorithm (step_jones_serial()).
	// If that algorithm detects (via last_j_f) that another algorithm has been used, it will reconstruct this table from scratch.
	bool* occupancy;
	int last_j_f;

	// Universal parameters
	// Grid size
	size_t w;
	size_t h;

	// Jeff Jones Physarum parameters
	float pop_dens;
	int diffK;
	float decayT;

	float SA;
	float RA;
	float SO;
	float SW;
	float SS;
	uint16_t depT;

	physarum2(size_t w, size_t h, float pop_dens) : c(), w(w), h(h), pop_dens(pop_dens) {
		cl_int err;

		c.build_program("physarum.cl");
		c.get_physarum_kernels();

		// Agents
		agents_n = (uint32_t) w*h*pop_dens;
		l_agents = (float*) malloc(agents_n * 3 * sizeof(float));

		r_agents = clCreateBuffer(c.c, CL_MEM_READ_WRITE, agents_n * 3 * sizeof(float), NULL, &err);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
		err = clRetainMemObject(r_agents);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		t_agents = clCreateBuffer(c.c, CL_MEM_READ_WRITE, agents_n * 3 * sizeof(float), NULL, &err);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
		err = clRetainMemObject(t_agents);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		// Deposit
		l_deposit = (uint16_t*) malloc(w * h * sizeof(uint16_t));

		r_deposit = clCreateBuffer(c.c, CL_MEM_READ_WRITE, w * h * sizeof(uint16_t), NULL, &err);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
		err = clRetainMemObject(r_deposit);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
		
		t_deposit = clCreateBuffer(c.c, CL_MEM_READ_WRITE, w * h * sizeof(uint16_t), NULL, &err);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
		err = clRetainMemObject(t_deposit);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		// Occupancy
		last_j_f = -1;
		occupancy = (bool*) malloc(w * h * sizeof(bool));

		// Set parameters to default values.
		diffK = 3;
		decayT = 0.1;
		SA = M_PI / 4;
		RA = M_PI / 4;
		SO = 8;
		SW = 1;
		SS = 2;
		depT = 100;
	}

	~physarum2() {
		clReleaseMemObject(r_agents);
		clReleaseMemObject(t_agents);
		clReleaseMemObject(r_deposit);
		clReleaseMemObject(t_deposit);

		free(l_agents);
		free(l_deposit);
	}

	// Fill the local arrays and write them to the OpenCL buffers.
	void initialize(int sd) {
		cl_int err;

		f = 0;

		for (int i = 0; i < w*h; i++) {
			l_deposit[i] = 0;
		}

		srand(sd);
		for (int i = 0; i < agents_n; i++) {
			l_agents[i*3  ] = (float) rand() / RAND_MAX * w;
			l_agents[i*3+1] = (float) rand() / RAND_MAX * h;
			l_agents[i*3+2] = (float) rand() / RAND_MAX * M_PI * 2;
		}

		write_deposit();
		write_agents();
	}

	// Call clSetKernelArg as needed to update the kernel arguments
	void update_args() {
		cl_int err;

		err = clSetKernelArg(c.k_blur, 0, sizeof(int), &w);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		err = clSetKernelArg(c.k_blur, 1, sizeof(int), &h);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		err = clSetKernelArg(c.k_blur, 2, sizeof(int), &diffK);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		err = clSetKernelArg(c.k_blur, 3, sizeof(float), &decayT);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		// Arguments 4 and 5 of blur() must be set on a per-execution basis, or the kernel will not be made aware of their swapping.
	}

	// Copy data from local buffer to remote buffer
	void write_deposit() {
		cl_int err;

		err = clEnqueueWriteBuffer(c.q, r_deposit, CL_TRUE, 0, w * h * sizeof(uint16_t), l_deposit, 0, NULL, NULL);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
	}

	void write_agents() {
		cl_int err;

		err = clEnqueueWriteBuffer(c.q, r_agents, CL_TRUE, 0, agents_n * 3 * sizeof(float), l_agents,  0, NULL, NULL);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
	}

	// Copy data from remote buffer to local buffer
	void read_deposit() {
		cl_int err;
		l_deposit[0] = 156;
		err = clEnqueueReadBuffer(c.q, r_deposit, CL_TRUE, 0, w * h * sizeof(uint16_t), l_deposit, 0, NULL, NULL);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
	}

	void read_agents() {
		cl_int err;

		err = clEnqueueReadBuffer(c.q, r_agents, CL_TRUE, 0, agents_n * 3 * sizeof(float), l_agents, 0, NULL, NULL);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
	}

	// hullo
	void hullo() {
		cl_int err;

		size_t gws = 5;
		err = clEnqueueNDRangeKernel(c.q, c.k_hullo, 1, NULL, &gws, NULL, 0, NULL, NULL);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);
	}

	// Apply box blur to the deposit map. box blur kernel width will be equal to this->diffK.
	void blur_evap() {
		cl_int err;

		err = clSetKernelArg(c.k_blur, 4, sizeof(cl_mem), &r_deposit);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		err = clSetKernelArg(c.k_blur, 5, sizeof(cl_mem), &t_deposit);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		size_t gws[] = {w, h};
		err = clEnqueueNDRangeKernel(c.q, c.k_blur, 2, NULL, gws, NULL, 0, NULL, NULL);
		if (err != CL_SUCCESS) printf("Error on %d: %d\n", __LINE__-1, err);

		cl_mem tmp = r_deposit;
		r_deposit = t_deposit;
		t_deposit = tmp;
	}

	// Step the simulation strictly according to the original (slow) rules of Jones's paper.
	void step_jones_serial() {
		// Test if the occupancy table needs to be reconstructed.
		if (last_j_f < f) {
			// Blank out the table.
			for (int i = 0; i < w*h; i++) {
				occupancy[i] = false;
			}

			// Write agent positions to the table.
			for (int i = 0; i < agents_n; i++) {
				int x = (int) l_agents[i*3  ];
				int y = (int) l_agents[i*3+1];
				occupancy[x + y*w] = true;
			}
		}

		f++;
		last_j_f = f;

		// Shuffle agents with Fisher-Yates shuffle.
		float temp;
		for (int i = agents_n-1; i > 0; i--) {
			int j = rand() % (i+1);

			temp = l_agents[i*3  ];
			l_agents[i*3  ] = l_agents[j*3  ];
			l_agents[j*3  ] = temp;

			temp = l_agents[i*3+1];
			l_agents[i*3+1] = l_agents[j*3+1];
			l_agents[j*3+1] = temp;
			
			temp = l_agents[i*3+2];
			l_agents[i*3+2] = l_agents[j*3+2];
			l_agents[j*3+2] = temp;
		}

		// Attempt to move all agents.
		for (int i = 0; i < agents_n; i++) {
			float nx = l_agents[i*3  ] + SS * cos(l_agents[i*3+2]);
			float ny = l_agents[i*3+1] + SS * sin(l_agents[i*3+2]);
			nx = fmod(nx + w, w);
			ny = fmod(ny + h, h);

			// Check for collision
			if (occupancy[(int) nx + ((int) ny) * w]) {
				l_agents[i*3+2] = (float) rand() / RAND_MAX * M_PI * 2;
				continue;
			}

			// Update occupancy table.
			occupancy[(int) l_agents[i*3] + ((int) l_agents[i*3+1]) * w] = false;
			occupancy[(int) nx + ((int) ny) * w] = true;

			// Move
			l_agents[i*3  ] = nx;
			l_agents[i*3+1] = ny;

			// Get sample positions.
			int Fx  = (int) (l_agents[i*3  ] + (SO * cos(l_agents[i*3+2])));
			int Fy  = (int) (l_agents[i*3+1] + (SO * sin(l_agents[i*3+2])));
			Fx = (Fx + w) % w;
			Fy = (Fy + h) % h;

			int FLx = (int) (l_agents[i*3  ] + (SO * cos(l_agents[i*3+2] + SA)));
			int FLy = (int) (l_agents[i*3+1] + (SO * sin(l_agents[i*3+2] + SA)));
			FLx = (FLx + w) % w;
			FLy = (FLy + h) % h;

			int FRx = (int) (l_agents[i*3  ] + (SO * cos(l_agents[i*3+2] - SA)));
			int FRy = (int) (l_agents[i*3+1] + (SO * sin(l_agents[i*3+2] - SA)));
			FRx = (FRx + w) % w;
			FRy = (FRy + h) % h;

			// Sample
			int F  = l_deposit[Fx  + Fy  * w];
			int FL = l_deposit[FLx + FLy * w];
			int FR = l_deposit[FRx + FRy * w];

			// Deposit chemoattractant
			l_deposit[(int) nx + ((int) ny) * w] += depT;

			if (F > FL && F > FR) {
				continue;
			}
			else if (F < FL && F < FR) {
				l_agents[i*3+2] += RA * (rand() % 2 - 1);
			}
			else if (FL < FR) {
				l_agents[i*3+2] -= RA;
			}
			else if (FR < FL) {
				l_agents[i*3+2] += RA;
			}
		}

		// Diffuse and evaporate chemoattractant.
		write_deposit();
		blur_evap();
		read_deposit();
	}
};
