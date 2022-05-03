.PHONY: clean
CUDIR     := /usr/local/nvidia

NVCC    := $(CUDIR)/bin/nvcc
CUFLAGS := -arch=compute_61 -code=sm_61,sm_61 -O3 --compiler-options -fopenmp

BIN     := heat_diffusion
LIB     :=

all: $(BIN)

$(BIN): $(BIN).cu
	$(NVCC) -o $(BIN) $(CUFLAGS) $(BIN).cu $(LIB)

clean:
	rm -f $(BIN)
