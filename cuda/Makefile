CXX := g++ 

CUDA_ROOT = /usr

CFLAGS = -Wall -I. -I$(CUDA_ROOT)/include
LIBS = -lcudart
LFLAGS = -L$(CUDA_ROOT)/lib $(LIBS) 

BIN = bandwidth_test_between2gpu

all: $(BIN)

bandwidth_test_between2gpu: bandwidth_test_between2gpu.cpp	

$(BIN):
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp, $^) $(LFLAGS)

.PHONY: clean all
clean:
	rm -rf $(BIN)	
