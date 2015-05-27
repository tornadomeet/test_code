CXX := mpicxx

CUDA_ROOT = /usr/local/cuda

CFLAGS = -Wall -I. -I$(CUDA_ROOT)/include
LIBS = -lcudart
LFLAGS = -L$(CUDA_ROOT)/lib64 $(LIBS) 

BIN = hello_world bandwidth_test_with_gpudirect

all: $(BIN)

hello_world: hello_world.cpp
bandwidth_test_with_gpudirect: bandwidth_test_with_gpudirect.cpp	

$(BIN):
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp, $^) $(LFLAGS)

.PHONY: clean all
clean:
	rm -rf $(BIN)	
