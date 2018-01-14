NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = chanvese
OBJ	        = sussman_kernel.o  chanvese.o 

default: $(EXE)

sussman_kernel.o: sussman_kernel.cu sussman_kernel.h 
	$(NVCC) -c -o $@ sussman_kernel.cu $(NVCC_FLAGS)


chanvese.o: ChanVese.cpp ChanVese.h sussman_kernel.cu sussman_kernel.h 
	$(NVCC) -c -o $@ ChanVese.cpp $(NVCC_FLAGS)


$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
