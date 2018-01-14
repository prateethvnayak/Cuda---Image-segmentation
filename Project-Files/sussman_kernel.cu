#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "sussman_kernel.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
//EPS 0.0001
#define BLOCK_SIZE 16
__global__ void sussman_kernel(float* phidd, float* phid, unsigned int height, unsigned int width);
__global__ void mean_kernel(float* phid, unsigned int* datain_d, unsigned int height, unsigned int width, unsigned int* sum_neg_d, unsigned int* sum_pos_d, int* c_neg_d, int* c_pos_d);
__global__ void force_kernel(float* phid, unsigned int* datain_d, float* F_d, unsigned int height, unsigned int width, double* maxF_d, int* stop_d, unsigned int* mean_neg_d, unsigned int* mean_pos_d, int* c_neg_d, int* c_pos_d);
//void kernel_call(float* phidd, float* phid, unsigned int height, unsigned int width);
//void kernel_call_mean(float* phid, float* datain_d, unsigned int height, unsigned int width, double* mean_neg_d, double* mean_pos_d, int* c_neg_d, int* c_pos_d);
//void kernel_call_force(float* phid, float* datain_d, float* F_d, unsigned int height, unsigned int width, double* maxF_d, int* stop_d, double mean_neg, double mean_pos);

__global__ void set(unsigned int* mean_neg_d, unsigned int* mean_pos_d, int* c_neg_d, int* c_pos_d, double* maxF_d, double* max_dphidt_d)
{
	int tid = threadIdx.x;
	if (tid == 0)
	{
		*mean_neg_d = 0;
		*mean_pos_d = 0;
		*c_neg_d = 0;
		*c_pos_d = 0;
		*maxF_d = 0;
		*max_dphidt_d = 0;
	}
}


__global__ void sussman_kernel_shared(float* phidd, float* phid, unsigned int height, unsigned int width)
{

	float d_phid;
	float a, b, c, d;

	int pixly = threadIdx.x + blockIdx.x * (BLOCK_SIZE-2);
	int pixlx = threadIdx.y + blockIdx.y * (BLOCK_SIZE-2);
	int iny = threadIdx.x + blockIdx.x * (BLOCK_SIZE-2) -1;
	int inx = threadIdx.y + blockIdx.y * (BLOCK_SIZE-2) -1;

        __shared__ float phi_s[BLOCK_SIZE][BLOCK_SIZE];
	
	inx = inx > 0 ? inx : width - 1;
	inx = inx < width ? inx : 0;
	iny = iny > 0 ? iny : height - 1;
	iny = iny < height ? iny : 0;		
	phi_s[threadIdx.x][threadIdx.y] = phid[inx*height + iny];	
	__syncthreads();

	if (pixlx < width && pixly < height)
	{

		int l_x = threadIdx.x;
		int r_x = threadIdx.x+2;
		int m_x = threadIdx.x+1;
		int u_y = threadIdx.y;
		int d_y = threadIdx.y+2;
		int m_y = threadIdx.y+1;
		

		if(threadIdx.x < BLOCK_SIZE-2 && threadIdx.y < BLOCK_SIZE-2 )
		{

			float sussman_sign = phi_s[m_x][m_y]/ sqrtf(phi_s[m_x][m_y]* phi_s[m_x][m_y]+ 1);
			if (phi_s[m_x][m_y]> 0)
			{
				a = fmax((float)(phi_s[m_x][m_y] - phi_s[l_x][m_y]), (float)0);
				b = fmin((float)(phi_s[r_x][m_y] - phi_s[m_x][m_y]), (float)0);
				c = fmax((float)(phi_s[m_x][m_y]- phi_s[m_x][d_y]), (float)0);
				d = fmin((float)(phi_s[m_x][u_y] - phi_s[m_x][m_y]), (float)0);

				d_phid = sqrtf(fmax(a*a, b*b) + fmax(c*c, d*d)) - 1;

			}
			else if (phi_s[m_x][m_y]< 0)
			{
				a = fmin((float)(phi_s[m_x][m_y]- phi_s[l_x][m_y]), (float)0);
				b = fmax((float)(phi_s[r_x][m_y] - phi_s[m_x][m_y]), (float)0);
				c = fmin((float)(phi_s[m_x][m_y]- phi_s[m_x][d_y]), (float)0);
				d = fmax((float)(phi_s[m_x][u_y] - phi_s[m_x][m_y]), (float)0);

				d_phid = sqrtf(fmax(a*a, b*b) + fmax(c*c, d*d)) - 1;

			}
			else
			{
				d_phid = 0;
			}
/* divergence elimination, worse performance
 
			int sign = 2*(phi_s[m_x][m_y]> 0)-1;
			float sussman_sign = phi_s[m_x][m_y]/ sqrtf(phi_s[m_x][m_y]* phi_s[m_x][m_y]+ 1);


				a = sign*fmax((float)sign* (phi_s[m_x][m_y] - phi_s[l_x][m_y]), (float)0);
				b = sign*fmin((float)(sign*phi_s[r_x][m_y] - phi_s[m_x][m_y]), (float)0);
				c = sign*fmax((float)(sign*phi_s[m_x][m_y]- phi_s[m_x][d_y]), (float)0);
				d = sign*fmin((float)(sign*phi_s[m_x][u_y] - phi_s[m_x][m_y]), (float)0);

				d_phid = sqrtf(fmax(a*a, b*b) + fmax(c*c, d*d)) - 1;

			
*/


			phidd[pixlx*height + pixly] = phi_s[threadIdx.x+1][threadIdx.y+1]- 0.5 * sussman_sign * d_phid;
		}
	}
}

__global__ void sussman_kernel(float* phidd, float* phid, unsigned int height, unsigned int width)
{

	
	int r_x, l_x, u_y, d_y;
	float d_phid;
	float a, b, c, d;

	int pixly = threadIdx.x + blockIdx.x * blockDim.x;
	int pixlx = threadIdx.y + blockIdx.y * blockDim.y;
	float plxly = phid[pixlx*height + pixly];
	if (pixlx < width && pixly < height)
	{
		// neighbor indices
		l_x = pixlx - 1 > 0 ? pixlx - 1 : width - 1;
		r_x = pixlx + 1 < width ? pixlx + 1 : 0;
		u_y = pixly - 1 > 0 ? pixly - 1 : height - 1;
		d_y = pixly + 1 < height ? pixly + 1 : 0;

		// sign of the value
		float sussman_sign = plxly / sqrtf(plxly * plxly + 1);

		// calculate difference between neghbor pixels (with thresholding)
		if (plxly > 0)
		{
			a = fmax((float)(plxly - phid[l_x*height + pixly]), (float)0);
			b = fmin((float)(phid[r_x*height + pixly] - plxly), (float)0);
			c = fmax((float)(plxly - phid[pixlx*height + d_y]), (float)0);
			d = fmin((float)(phid[pixlx*height + u_y] - plxly), (float)0);

			d_phid = sqrtf(fmax(a*a, b*b) + fmax(c*c, d*d)) - 1;

		}
		else if (plxly < 0)
		{
			a = fmin((float)(plxly - phid[l_x*height + pixly]), (float)0);
			b = fmax((float)(phid[r_x*height + pixly] - plxly), (float)0);
			c = fmin((float)(plxly - phid[pixlx*height + d_y]), (float)0);
			d = fmax((float)(phid[pixlx*height + u_y] - plxly), (float)0);

			d_phid = sqrtf(fmax(a*a, b*b) + fmax(c*c, d*d)) - 1;

		}
		else
		{
			d_phid = 0;
		}
		// update phi
		phidd[pixlx*height + pixly] = plxly - 0.5 * sussman_sign * d_phid;
	}
}

void sussman_kernel_call(float* phidd, float* phid, unsigned int height, unsigned int width)
{
/*
       //shared memory version
	dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridsize((height + BLOCK_SIZE-1) / (BLOCK_SIZE-2), (width + BLOCK_SIZE-1) / (BLOCK_SIZE-2), 1);
	// kernel call
	sussman_kernel_shared << <gridsize, blocksize >> >(phidd, phid, height, width);
*/
        //global memory version
	dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridsize((height + BLOCK_SIZE-1) / BLOCK_SIZE, (width + BLOCK_SIZE-1) / BLOCK_SIZE, 1);
	// kernel call
	sussman_kernel << <gridsize, blocksize >> >(phidd, phid, height, width);


}

__device__ double atomicAddDouble(double* address,
	double val) {
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old =
		*address_as_ull, assumed;
	do {
		assumed = old;	// READ
		old = atomicCAS(address_as_ull, assumed,
			val + assumed);	// MODIFY + WRITE
	} while (assumed != old);
	return old;
}


__global__ void mean_kernel(float* phid, unsigned int* datain_d, unsigned int height, unsigned int width, unsigned int* sum_neg_d, unsigned int* sum_pos_d, int* c_neg_d, int* c_pos_d)
{
	__shared__ unsigned int s_sum_neg[32];
	__shared__ unsigned int s_sum_pos[32];
	__shared__ int s_c_neg[32];
	__shared__ int s_c_pos[32];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;
	// private copies of sum and counters
	if (tx < 32)
	{
		s_sum_neg[tx] = 0;
		s_sum_pos[tx] = 0;
		s_c_neg[tx] = 0;
		s_c_pos[tx] = 0;
	}
	__syncthreads();
	// calculate sum and count
	if (tid < height*width)
	{
		if (phid[tid] <= 0)
		{
			atomicAdd(&s_sum_neg[tx%32], datain_d[tid]);
			atomicAdd(&s_c_neg[tx%32], 1);
		}
		else
		{
			atomicAdd(&s_sum_pos[tx%32], datain_d[tid]);
			atomicAdd(&s_c_pos[tx%32], 1);
		}

		__syncthreads();
		
		if(tx<32)
		{
			atomicAdd(&s_sum_neg[0], s_sum_neg[tx]);
			atomicAdd(&s_sum_pos[0], s_sum_pos[tx]);
			atomicAdd(&s_c_neg[0], s_c_neg[tx]);
			atomicAdd(&s_c_pos[0], s_c_pos[tx]);


		}

		if (tx == 0)
		{
			// accumulate global sum and counter
			atomicAdd(sum_neg_d, s_sum_neg[0]);
			atomicAdd(c_neg_d, s_c_neg[0]);
			atomicAdd(sum_pos_d, s_sum_pos[0]);
			atomicAdd(c_pos_d, s_c_pos[0]);
		}
	}
}

//For calculation of mean
void kernel_call_mean(float* phid, unsigned int* datain_d, unsigned int height, unsigned int width, unsigned int* mean_neg_d, unsigned int* mean_pos_d, int* c_neg_d, int* c_pos_d, double* maxF_d, double* max_dphidt_d)
{
	dim3 blocksize(256, 1, 1);
	dim3 gridsize((height*width + 255) / 256, 1, 1);
	// kernel call to set
	set << <1, 1 >> > (mean_neg_d, mean_pos_d, c_neg_d, c_pos_d, maxF_d, max_dphidt_d);

	//kernel for mean
	mean_kernel << <gridsize, blocksize >> >(phid, datain_d, height, width, mean_neg_d, mean_pos_d, c_neg_d, c_pos_d);
	// copy back to host
}

__device__ static double atomicMaxDouble(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void force_kernel(float* phid, unsigned int* datain_d, float* F_d, unsigned int height, unsigned int width, double* maxF_d, int* stop_d, unsigned int* mean_neg_d, unsigned int* mean_pos_d, int* c_neg_d, int* c_pos_d)
{
	__shared__ double s_maxF[32];
	__shared__ float mean_neg;
	__shared__ float mean_pos;
	//__shared__ int s_stop;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;
	// mean of inside and outside of 0 level set
	if (tx == 0)
	{
		mean_neg = *mean_neg_d / ((float)(*c_neg_d) + 0.00001);
		mean_pos = *mean_pos_d / ((float)(*c_pos_d) + 0.00001);
		
	}
	if (tx < 32)
	{
		s_maxF[tx] = 0;
	}
	__syncthreads();

	float phiVal = phid[tid];
	float dataVal = datain_d[tid];
	if (tid < height*width)
	{
		// calculate the force pushing the boundary
		float val = 0.0f;
		if (phiVal < 1.2 && phiVal > -1.2)
		{
			val = (dataVal - mean_neg) * (dataVal - mean_neg) - (dataVal - mean_pos) * (dataVal - mean_pos);
			atomicMaxDouble(&s_maxF[tx%32], fabs(val));
			F_d[tid] = val;

		}

		__syncthreads();
		if (tx < 32)
		{
			atomicMaxDouble(&s_maxF[0], s_maxF[tx]);
		}
	//	__syncthreads();
		if (tx == 0)
		{
			atomicMaxDouble(maxF_d, fabs(s_maxF[0]));
		}
	}
}
void kernel_call_force(float* phid, unsigned int* datain_d, float* F_d, unsigned int height, unsigned int width, double* maxF_d, int* stop_d, unsigned int* mean_neg_d, unsigned int* mean_pos_d, int* c_neg_d, int* c_pos_d)
{

	dim3 blocksize(256, 1, 1);
	dim3 gridsize((height*width + 255) / 256, 1, 1);
	// kernel call
	force_kernel << <gridsize, blocksize >> >(phid, datain_d, F_d, height, width, maxF_d, stop_d, mean_neg_d, mean_pos_d, c_neg_d, c_pos_d);
	// copy back to host
	//                cudaDeviceSynchronize();
}

__global__ void gradient_kernel(float* phid, float* curvature_d, float* F_d, float* dphidt_d, double* max_dphidt_d, double alpha, double* maxF_d, unsigned int height, unsigned int width)
{
	__shared__ double max_dphidt_s;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;

	if (tx == 0)
		max_dphidt_s = 0;

	__syncthreads();


	if (tid < height*width)
	{
		// calculate gradient only near boundary of 0 level set
		if (phid[tid] < 1.2 && phid[tid] > -1.2)
		{
			dphidt_d[tid] = (F_d[tid] / *maxF_d) + alpha * curvature_d[tid];
			atomicMaxDouble(&max_dphidt_s, (double)dphidt_d[tid]);
		}

		__syncthreads();

		// max of gradient for normalization
		if (tx == 0)
		{
			atomicMaxDouble(max_dphidt_d, max_dphidt_s);
		}
	}

}

// Function definition to launch the kernel to compute the gradient
void gradient_kernel_call(float* phid, float* curvature_d, float* F_d, float* dphidt_d, unsigned int height, unsigned int width, double* max_dphidt_d, double alpha, double* maxF_d)
{
	dim3 blocksize(256, 1, 1);
	dim3 gridsize((height*width + 255) / 256, 1, 1);
	gradient_kernel << <gridsize, blocksize >> >(phid, curvature_d, F_d, dphidt_d, max_dphidt_d, alpha, maxF_d, height, width);
}

__global__ void CFL_kernel(float* phidCFL, float* phid, float* dphidt_d, double* max_dphidt_d, unsigned int height, unsigned int width)
{
	__shared__ float dt;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// calculate step size
	if (threadIdx.x == 0)
	{
		dt = 0.45 / (*max_dphidt_d + 0.00001);
	}
	
	__syncthreads();
	if (tid < height*width)
	{
		// update only pixels near boundary of 0 level set
		if (phid[tid] < 1.2 && phid[tid] > -1.2)
		{
			phidCFL[tid] = phid[tid] + dt * dphidt_d[tid];
		}
		else
		{
			phidCFL[tid] = phid[tid];
		}

	}
}

// Function definition to launch the kernel for CFL condition
void CFL_kernel_call(float* phidCFL, float* phid, float* dphidt_d, unsigned int height, unsigned int width, double *max_dphidt_d)
{
	dim3 blocksize(256, 1, 1);
	dim3 gridsize((height*width + 255) / 256, 1, 1);
	CFL_kernel << <gridsize, blocksize >> >(phidCFL, phid, dphidt_d, max_dphidt_d, height, width);
}

__global__ void curvature_kernel(float* curvature_d, float* phid, unsigned int height, unsigned int width)
{
	float phi_x, phi_y, phi_xx, phi_yy, phi_xy;
	int xm1, ym1, xp1, yp1;

	int tid_y = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_x = threadIdx.y + blockIdx.y * blockDim.y;

	if (tid_x<width&&tid_y<height)
	{
		if (phid[tid_x*height + tid_y] < 1.2 && phid[tid_x*height + tid_y] > -1.2)
		{
			// boundary check
			xm1 = tid_x - 1 < 0 ? 0 : tid_x - 1;
			ym1 = tid_y - 1 < 0 ? 0 : tid_y - 1;
			xp1 = tid_x + 1 >= width ? width - 1 : tid_x + 1;
			yp1 = tid_y + 1 >= height ? height - 1 : tid_y + 1;
			

			// differences between neighbors
			phi_x = -phid[xm1*height + tid_y] + phid[xp1*height + tid_y];
			phi_y = -phid[tid_x*height + ym1] + phid[tid_x*height + yp1];
			phi_xx = phid[xm1*height + tid_y] + phid[xp1*height + tid_y] - 2 * phid[tid_x*height + tid_y];
			phi_yy = phid[tid_x*height + ym1] + phid[tid_x*height + yp1] - 2 * phid[tid_x*height + tid_y];
			phi_xy = 0.25*(-phid[xm1*height + ym1] - phid[xp1*height + yp1] + phid[xp1*height + ym1] + phid[xm1*height + yp1]);
			// curvature calculation
			curvature_d[tid_x*height + tid_y] = phi_x*phi_x * phi_yy + phi_y*phi_y * phi_xx - 2 * phi_x * phi_y * phi_xy;
			curvature_d[tid_x*height + tid_y] = curvature_d[tid_x*height + tid_y] / (phi_x*phi_x + phi_y*phi_y + 0.00001);
		}
		else
		{
			curvature_d[tid_x*height + tid_y] = 0;
		}
	}
}


__global__ void curvature_kernel_shared(float* curvature_d, float* phid, unsigned int height, unsigned int width)
{
	float phi_x, phi_y, phi_xx, phi_yy, phi_xy;
	//int xm1, ym1, xp1, yp1;

        __shared__ float phi_s[BLOCK_SIZE][BLOCK_SIZE];

	
	int tid_y = threadIdx.x + blockIdx.x * (BLOCK_SIZE-2);
	int tid_x = threadIdx.y + blockIdx.y * (BLOCK_SIZE-2);    
	int iny = threadIdx.x + blockIdx.x * (BLOCK_SIZE-2) -1;
	int inx = threadIdx.x + blockIdx.x * (BLOCK_SIZE-2) -1;

	int l_x, r_x, m_x, u_y, d_y, m_y;
	
	inx = inx > 0 ? inx : 0;
	inx = inx < width ? inx : width - 1;
	iny = iny > 0 ? iny : 0;
	iny = iny < height ? iny : height - 1;
		
	phi_s[threadIdx.x][threadIdx.y] = phid[inx*height + iny];	
	__syncthreads();


	if (tid_x<width&&tid_y<height)
	{
		l_x = threadIdx.x;
		r_x = threadIdx.x+2;
		m_x = threadIdx.x+1;
		u_y = threadIdx.y;
		d_y = threadIdx.y+2;
		m_y = threadIdx.y+1;
		if (phi_s[m_x][m_y] < 1.2 && phi_s[m_x][m_y] > -1.2)
		{
			// boundary check


			// differences between neighbors
			phi_x = -phi_s[l_x][m_y] + phi_s[r_x][m_y];
			phi_y = -phi_s[m_x][u_y] + phi_s[m_x][d_y];
			phi_xx = phi_s[l_x][m_y] + phi_s[r_x][m_y] - 2 * phi_s[m_x][m_y];
			phi_yy = phi_s[m_x][u_y] + phi_s[m_x][d_y] - 2 * phi_s[m_x][m_y];
			phi_xy = 0.25*(-phi_s[l_x][u_y] - phi_s[r_x][d_y] + phi_s[r_x][u_y] + phi_s[l_x][d_y]);
			// curvature calculation
			curvature_d[tid_x*height + tid_y] = phi_x*phi_x * phi_yy + phi_y*phi_y * phi_xx - 2 * phi_x * phi_y * phi_xy;
			curvature_d[tid_x*height + tid_y] = curvature_d[tid_x*height + tid_y] / (phi_x*phi_x + phi_y*phi_y + 0.00001);
		}
		else
		{
			curvature_d[tid_x*height + tid_y] = 0;
		}
	}
}


void curvature_kernel_call(float* curvature_d, float* phid, unsigned int height, unsigned int width)
{
/*
	// global memory version
	dim3 blocksize(16, 16, 1);
	dim3 gridsize((height + 15) / 16, (width + 15) / 16, 1);
	// kernel call
	curvature_kernel << <gridsize, blocksize >> >(curvature_d, phid, height, width);
*/	
	// shared memory version
	dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridsize((height + BLOCK_SIZE-1) / (BLOCK_SIZE-2), (width + BLOCK_SIZE-1) / (BLOCK_SIZE-2), 1);

	// kernel call
	curvature_kernel_shared << <gridsize, blocksize >> >(curvature_d, phid, height, width);

	

}