.PHONY: clean
CUDIR     := /usr/local/nvidia

NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61,sm_61 -Xptxas -v

BIN     := poisson
LIB     :=

all: $(BIN)

$(BIN): $(BIN).cu
	$(NVCC) -o $(BIN) $(NVFLAGS) $(BIN).cu $(LIB)

clean:
	rm -f $(BIN) *.dat
