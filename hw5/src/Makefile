.PHONY: clean
CUDIR     := /usr/local/nvidia

NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61,sm_61 -Xptxas -v

BIN1     := hist_1gpu_gmem
BIN2     := hist_1gpu_shmem
LIB     :=

all: $(BIN1) $(BIN2)

$(BIN1): $(BIN1).cu
	$(NVCC) -o $(BIN1) $(NVFLAGS) $(BIN1).cu $(LIB)

$(BIN2): $(BIN2).cu
	$(NVCC) -o $(BIN2) $(NVFLAGS) $(BIN2).cu $(LIB)

clean:
	rm -f $(BIN1) $(BIN2) *.dat
