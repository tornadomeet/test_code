CXX := g++ 

OPENCV_ROOT = /mnt/disk/0/multimedia/hzwuw2014/tools/opencv
OPENCV_INCLUDE = $(OPENCV_ROOT)/include
OPENCV_LIB = $(OPENCV_ROOT)/lib

CFLAGS = -Wall -I. -I$(OPENCV_INCLUDE)
LIBS = -lopencv_highgui -lopencv_core -lopencv_imgproc
LFLAGS = $(LIBS) -L$(OPENCV_LIB) 

BIN = resize_directory_imgs 

all: $(BIN)

resize_directory_imgs: resize_directory_imgs.cpp

$(BIN):
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp, $^) $(LFLAGS)

.PHONY: clean all
clean:
	rm -rf $(BIN)
