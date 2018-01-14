#pragma once


class ChanVese
	{
	public:
		unsigned int *datain, *dataout;
		int *init_mask;
		float *phi;
		float *dist_to_nonzero;
		float *dist_to_zero;
		float *phi_s;
		float *F;
		float *curvature;
		float *dphidt;
		int width, height;

		//bool SendDataFromCS(unsigned int *datain, int width, int height);
		//bool ReceiveDataFromCPP(unsigned int *datain, int width, int height);
		//void PerformThreshold(unsigned int Threshold);
		void chanvese_edge_parallel();
		void chanvese_edge_sequential();
    int ReadParamsFile(char* file_name);
    int ReadFile(char* file_name);
    void WriteFile(char* file_name);

	};
