#ifndef OPT_KERNEL
#define OPT_KERNEL
#include <cuda.h>

//void sussman_kernel(float* phid, float* phidd);
void sussman_kernel_call(float* phidd, float* phid, unsigned int height, unsigned int width);
//void mean_kernel(float* phid, float* datain_d, unsigned int height, unsigned int width, double* mean_neg_d, double* mean_pos_d, int* c_neg_d, int* c_pos_d);
void kernel_call_mean(float* phid, unsigned int* datain_d, unsigned int height, unsigned int width, unsigned int* mean_neg_d, unsigned int* mean_pos_d, int* c_neg_d, int* c_pos_d, double* maxF_d, double* max_dphidt_d);
//void force_kernel(float* phid, float* datain_d, float* F_d, unsigned int height, unsigned int width, double* maxF_d, int* stop_d, double mean_neg, double mean_pos);
void kernel_call_force(float* phid, unsigned int * datain_d, float* F_d, unsigned int height, unsigned int width, double* maxF_d, int* stop_d, unsigned int* mean_neg_d, unsigned int* mean_pos_d, int* c_neg_d, int* c_pos_d);

void gradient_kernel_call(float* phid, float* curvature_d, float* F_d, float* dphidt_d, unsigned int height, unsigned int width, double* max_dphidt_d, double alpha, double* maxF_d);

void CFL_kernel_call(float* phidCFL, float* phid, float* dphidt_d, unsigned int height, unsigned int width, double* max_dphidt_d);
void curvature_kernel_call(float* curvature_d,float* phid, unsigned int height, unsigned int width);

#endif
