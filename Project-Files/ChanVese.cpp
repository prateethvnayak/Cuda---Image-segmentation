#include "ChanVese.h"
#include <memory.h>
#include <math.h>
#include <sys/time.h>
#include "sussman_kernel.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#define EPS 0.00001



int ChanVese::ReadParamsFile(char* file_name)
{
    FILE* input = fopen(file_name, "r");
    fscanf(input, "%d", &width);
    fscanf(input, "%d", &height);
    return 0;
}

int ChanVese::ReadFile(char* file_name)
{
    unsigned int data_read = width * height;
    FILE* input = fopen(file_name, "r");
    unsigned i = 0;
    char line[10];
    for (i = 0; i < data_read; i++)
    {
          //fscanf(input, "%d\n", &(datain[i]));
        fgets(line, 10, input);
	datain[i]=atoi(line);
    }
    return data_read;
}

void ChanVese::WriteFile(char* file_name)
{

    unsigned int size = width * height;
    FILE* output = fopen(file_name, "w");

    for (unsigned i = 0; i < size; i++) {
        fprintf(output, "%d\n", dataout[i]);
    }
}



void ChanVese::chanvese_edge_sequential()
{
ChanVese::phi = new float[width*height];


	int max_its = 1000;
	double alpha = 0.2;
	int thresh = 0;
	float distance;
	int x_center = width / 2;
	int y_center = height / 2;
	int r = fmin(width, height) / 4;

	if( width * height > 1000000)
	{
		max_its = width * height/1000;
		if(max_its > 4000)
		{
			max_its = 4000;
		}
	}

	
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			distance = sqrt((i - x_center)*(i - x_center) + (j - y_center)*(j - y_center));

			phi[i*height + j] = (distance - r) + signbit(distance - r) - 0.5;

		}
	}

	ChanVese::init_mask = new int[width*height];


	int its = 0;
	int stop = 0;
	int* prev_mask = init_mask;
	int	c = 0;

	// mask to phi

	ChanVese::dist_to_nonzero = new float[width*height];
	ChanVese::dist_to_zero = new float[width*height];
	
	
	ChanVese::F = new float[width*height];
	ChanVese::dphidt = new float[width*height];
	ChanVese::curvature = new float[width*height];
	ChanVese::phi_s = new float[width*height];

	
	// start 
	double mean_neg = 0;
	double mean_pos = 0;
	unsigned int sum_neg = 0;
	unsigned int sum_pos = 0;
	int c_neg = 0;
	int c_pos = 0;
	double max_F = 0;
	double max_dphidt = 0;

	// device variables
	float *phidd;
	float *phid;
	float *dphidt_d;
	float *curvature_d;
	double *max_dphidt_d;

        //For mean calculation
        unsigned int *datain_d;
	unsigned int *mean_neg_d;
	unsigned int *mean_pos_d;
	int *c_neg_d;
	int *c_pos_d;

        float *F_d;
        double *maxF_d;
        int *stop_d;


	while (its < max_its && ~stop)
	{



		mean_neg = 0;
		mean_pos = 0;
		c_neg = 0;
		c_pos = 0;
		// interior and exterior mean
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				if (phi[i*height + j] <= 0)
				{
					mean_neg += datain[i*height + j];
					c_neg += 1;
				}
				else
				{
					mean_pos += datain[i*height + j];
					c_pos += 1;
				}
			}
		}
		mean_neg = mean_neg / ((double)c_neg + EPS);
		mean_pos = mean_pos / ((double)c_pos + EPS);
	
		stop = 1;
		max_F = 0;





		// Force from image information
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				if (phi[i*height + j] < 1.2 && phi[i*height + j] > -1.2)
				{
					stop = 0;
					F[i*height + j] = (datain[i*height + j] - mean_neg) * (datain[i*height + j] - mean_neg) - (datain[i*height + j] - mean_pos) * (datain[i*height + j] - mean_pos);
					if (max_F < fabs(F[i*height + j]))
					{
						max_F = fabs(F[i*height + j]);
					}
				}
				else
				{
//					F[i*height + j] = 0;
				}
			}
		}

	


	
		// Force from curvature penalty
		float phi_x, phi_y, phi_xx, phi_yy, phi_xy;
		int xm1, ym1, xp1, yp1;
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				if (phi[x*height + y] < 1.2 && phi[x*height + y] > -1.2)
				{
					xm1 = x - 1 < 0 ? 0 : x - 1;
					ym1 = y - 1 < 0 ? 0 : y - 1;
					xp1 = x + 1 >= width ? width - 1 : x + 1;
					yp1 = y + 1 >= height ? height - 1 : y + 1;

					phi_x = -phi[xm1*height + y] + phi[xp1*height + y];
					phi_y = -phi[x*height + ym1] + phi[x*height + yp1];
					phi_xx = phi[xm1*height + y] + phi[xp1*height + y] - 2 * phi[x*height + y];
					phi_yy = phi[x*height + ym1] + phi[x*height + yp1] - 2 * phi[x*height + y];
					phi_xy = 0.25*(-phi[xm1*height + ym1] - phi[xp1*height + yp1] + phi[xp1*height + ym1] + phi[xm1*height + yp1]);

					curvature[x*height + y] = phi_x*phi_x * phi_yy + phi_y*phi_y * phi_xx - 2 * phi_x * phi_y * phi_xy;
					curvature[x*height + y] = curvature[x*height + y] / (phi_x*phi_x + phi_y*phi_y + EPS);
				}
				else
				{
					curvature[x*height + y] = 0;
				}
			}
		}

        

		

		// Gradient descent to minimize energy
	     max_dphidt = 0;
		float dt;



		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				if (phi[x*height + y] < 1.2 && phi[x*height + y] > -1.2)
				{
					dphidt[x*height + y] = F[x*height + y] / max_F + alpha * curvature[x*height + y];
					if (max_dphidt < fabs(dphidt[x*height + y]))
					{
						max_dphidt = fabs(dphidt[x*height + y]);
					}
				}
				
			}
		}

		// Maintain the CFL condition
		dt = 0.45 / (max_dphidt + EPS);


		// CLF kernel end

                // Evolve the curve
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				if (phi[x*height + y] < 1.2 && phi[x*height + y] > -1.2)
				{
					phi[x*height + y] += dt * dphidt[x*height + y];
				}
			}
		} 
                

	
				// Level set re-initialization by the sussman method
		// sussman
		int r_x, l_x, u_y, d_y;
		float sussman_dt = 0.5;
		float d_phi;
		float a_p, a_n, b_p, b_n, c_p, c_n, d_p, d_n;
		float sussman_sign;
		
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				l_x = x - 1 > 0 ? x - 1 : width - 1;
				r_x = x + 1 < width ? x + 1 : 0;
				u_y = y - 1 > 0 ? y - 1 : height - 1;
				d_y = y + 1 < height ? y + 1 : 0;
				if (phi[x*height + y] > 0)
				{
					a_p = fmax(phi[x*height + y] - phi[l_x*height + y], 0);
					b_n = fmin(phi[r_x*height + y] - phi[x*height + y], 0);
					c_p = fmax(phi[x*height + y] - phi[x*height + d_y], 0);
					d_n = fmin(phi[x*height + u_y] - phi[x*height + y], 0);
					d_phi = sqrt(fmax(a_p*a_p, b_n*b_n) + fmax(c_p*c_p, d_n*d_n)) - 1;
					sussman_sign = phi[x*height + y] / sqrt(phi[x*height + y] * phi[x*height + y] + 1);
					phi_s[x*height + y] = phi[x*height + y] - sussman_dt * sussman_sign * d_phi;
				}
				else if (phi[x*height + y] < 0)
				{
					a_n = fmin(phi[x*height + y] - phi[l_x*height + y], 0);
					b_p = fmax(phi[r_x*height + y] - phi[x*height + y], 0);
					c_n = fmin(phi[x*height + y] - phi[x*height + d_y], 0);
					d_p = fmax(phi[x*height + u_y] - phi[x*height + y], 0);
					d_phi = sqrt(fmax(a_n*a_n, b_p*b_p) + fmax(c_n*c_n, d_p*d_p)) - 1;
					sussman_sign = phi[x*height + y] / sqrt(phi[x*height + y] * phi[x*height + y] + 1);
					phi_s[x*height + y] = phi[x*height + y] - sussman_dt * sussman_sign * d_phi;
				}
				else 
				{
					phi_s[x*height + y] = 0;
				}
			}
		}

		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				phi[x*height + y] = phi_s[x*height + y];
			}
		}

	



	
		//congergence to be filled
		its += 1;

	}


		
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			dataout[i*height + j] = (phi[i*height + j] < 2 && phi[i*height + j] > -2 ) ? 255 : 0;
		//	dataout[i*height + j] = (phi[i*height + j] <= 0 ) ? 255 : 0;

		}
	}
}


void ChanVese::chanvese_edge_parallel()
{
ChanVese::phi = new float[width*height];


	int max_its = 1000;
	double alpha = 0.2;
	int thresh = 0;
	float distance;
	int x_center = width / 2;
	int y_center = height / 2;
	int r = fmin(width, height) / 4;
	if( width * height > 1000000)
	{
		max_its = width * height/1000;
		if(max_its > 4000)
		{
			max_its = 4000;
		}
	}
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			distance = sqrt((i - x_center)*(i - x_center) + (j - y_center)*(j - y_center));

			phi[i*height + j] = (distance - r) + signbit(distance - r) - 0.5;

		}
	}

	ChanVese::init_mask = new int[width*height];


	int its = 0;
	int stop = 0;
	int* prev_mask = init_mask;
	int	c = 0;

	// mask to phi

	ChanVese::dist_to_nonzero = new float[width*height];
	ChanVese::dist_to_zero = new float[width*height];
	
	
	ChanVese::F = new float[width*height];
	ChanVese::dphidt = new float[width*height];
	ChanVese::curvature = new float[width*height];
	ChanVese::phi_s = new float[width*height];

	
	// start 
	double mean_neg = 0;
	double mean_pos = 0;
	unsigned int sum_neg = 0;
	unsigned int sum_pos = 0;
	int c_neg = 0;
	int c_pos = 0;
	double max_F = 0;
	double max_dphidt = 0;

	// device variables
	float *phidd;
	float *phid;
	float *dphidt_d;
	float *curvature_d;
	double *max_dphidt_d;
	cudaMalloc((void**)&max_dphidt_d, sizeof(double));
	cudaMalloc((void**)&dphidt_d, height*width * sizeof(float));
	cudaMalloc((void**)&curvature_d, height * width * sizeof(float));
	cudaMalloc((void**)&phidd, height * width * sizeof(float));
	cudaMalloc((void**)&phid, height * width * sizeof(float));

        //For mean calculation
        unsigned int *datain_d;
	unsigned int *mean_neg_d;
	unsigned int *mean_pos_d;
	int *c_neg_d;
	int *c_pos_d;
	cudaMalloc((void**)&datain_d, height * width * sizeof(unsigned int));
	cudaMemcpy(datain_d, datain, height*width * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&mean_neg_d, sizeof(unsigned int));
	cudaMalloc((void**)&mean_pos_d, sizeof(unsigned int));
	cudaMalloc((void**)&c_neg_d, sizeof(int));
	cudaMalloc((void**)&c_pos_d, sizeof(int));

        float *F_d;
        double *maxF_d;
        int *stop_d;
        cudaMalloc((void**)&F_d, height * width * sizeof(float));
	cudaMalloc((void**)&maxF_d, sizeof(double));
	cudaMalloc((void**)&stop_d, sizeof(int));


//	struct timeval t1, t2;
//	double elapsedTime;

	// start timer
//	gettimeofday(&t1, NULL);
	cudaMemcpy(phidd, phi, height*width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(phid, phi, height*width * sizeof(float), cudaMemcpyHostToDevice);

	
	while (its < max_its && ~stop)
	{



               // kernel mean start

                kernel_call_mean(phidd,datain_d,height,width,mean_neg_d,mean_pos_d,c_neg_d,c_pos_d, maxF_d,max_dphidt_d);

               // kernel mean finish



	      // kernel force start 

           
                kernel_call_force(phidd, datain_d,F_d, height, width, maxF_d, stop_d, mean_neg_d, mean_pos_d, c_neg_d, c_pos_d);


              // kernel force finish



		// Gradient descent to minimize energy


                // gradient kernel start

                gradient_kernel_call(phidd, curvature_d, F_d, dphidt_d, height, width, max_dphidt_d, 0.2, maxF_d); 
                           
                // gradient kernel end


		// Maintain the CFL condition

                //CLF kernel start

                CFL_kernel_call(phid, phidd, dphidt_d, height, width, max_dphidt_d); // Function call to launch the kernel


		// CLF kernel end


	

	// kernel sussman start
		sussman_kernel_call(phidd,phid,height,width);
        // kernel sussman finish

		//congergence to be filled
		its += 1;

	}
	// sussman kernel call
	cudaMemcpy(phi, phid, height*width * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    
		
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			dataout[i*height + j] = (phi[i*height + j] < 2 && phi[i*height + j] > -2) ? 255 : 0;
		//	dataout[i*height + j] = (phi[i*height + j] <= 0) ? 255 : 0;

		}
	}
}


int main()
{
	ChanVese p1;
	p1.ReadParamsFile("size.txt");

	p1.datain = (unsigned int*)malloc(p1.height * p1.width * sizeof(unsigned int));
	p1.dataout = (unsigned int*)malloc(p1.height * p1.width * sizeof(unsigned int));

  p1.ReadFile("data.txt");

  struct timeval t1, t2;
  double elapsedTime;
  cudaFree(0);
  gettimeofday(&t1, NULL);
  // start parallel timer
  p1.chanvese_edge_parallel();
  // end parallel timer
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  printf("Chanvese parallel time: %f\n", elapsedTime);
  p1.WriteFile("out_gpu.txt");

  ChanVese p2;
  p2.ReadParamsFile("size.txt");

  p2.datain = (unsigned int*)malloc(p2.height * p2.width * sizeof(unsigned int));
  p2.dataout = (unsigned int*)malloc(p2.height * p2.width * sizeof(unsigned int));


  
  p2.ReadFile("data.txt");
  gettimeofday(&t1, NULL);
  // start sequential timer
  p2.chanvese_edge_sequential();
  // end sequential timer
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  printf("Chanvese sequential time: %f\n", elapsedTime);

  p2.WriteFile("out_seq.txt");
  return 0;
}
