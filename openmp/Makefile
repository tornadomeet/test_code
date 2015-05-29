CXX := g++ 
CUXX := nvcc 

CUDA_ROOT = /usr/local/cuda

CFLAGS = -Wall -fopenmp -I. -I$(CUDA_ROOT)/include
CUFLAGS = -Xcompiler -fopenmp -I. -I$(CUDA_ROOT)/include
LIBS = -lcudart
LFLAGS = -L$(CUDA_ROOT)/lib $(LIBS) 

BIN = hello_world
CUBIN = openmp_build_with_cuda_test 

all: $(BIN) $(CUBIN)

hello_world: hello_world.cpp
openmp_build_with_cuda_test: openmp_build_with_cuda_test.cu	

$(BIN):
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp, $^) $(LFLAGS)
$(CUBIN):
	$(CUXX) -ccbin $(CXX) -m64 $(CUFLAGS) -o $@ $(filter %.cpp %.cu, $^) $(LFLAGS)

.PHONY: clean all
clean:
	rm -rf $(BIN) $(CUBIN)
