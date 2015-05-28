/*	
	the bin is used for testing the bandwith between 2 gpus, the usage is:
	bandwidth_test_between2gpu 0 1
	but there is one problem troubled me: the cudaMemcpy in K80(enabled P2P) is asynchronization!?
	when we use one K80 for test, which has 2 gpu in one board. in the code, let set use_cuda_time = false, means
	use getSystemTime() for counting the passed time. if when don't set the funtion cudaDeviceEnablePeerAccess(),
	then 2gpu in K80 is not P2P, and the cudaMemcpy() is all going well. but  when we set cudaDeviceEnablePeerAccess() on, 
	then cudaMemcpy() is asynchronization, because the count time is nearly zero, which is very surprising! if it's true,
	then cudaMemcpy() and cudaMemcpyAsync() is the same between 2gpu which can P2P and be seted as P2P.
*/

#include <iostream>
#include <string>
#include <stdlib.h>
#include<sys/timeb.h>
#include <cuda_runtime.h>

#define WIDTH 10000
#define HEIGHT 10000
#define CNT 10 

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        std::cerr << "CUDA error calling \""#call"\", code is " << err << std::endl; \
        }

// get the system time right now
long long getSystemTime() 
{
        struct timeb t;
        ftime(&t);
        return 1000 * t.time + t.millitm;
}

// Initialize an array with random data (between 0 and 1)
void init_data(float *data, int dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {   
        data[i] = (float)rand() / RAND_MAX;
    }   
}

// use_cuda_time = 1: use cudaEventElapsedTime()
// or use getSystemTime()
void test_2gpu(float *d_send_data, float *d_recv_data, int size, int id0, int id1, bool use_cuda_time)
{
	if(use_cuda_time) {
		cudaEvent_t start_event, stop_event;
		float time_memcpy;
		int eventflags = cudaEventBlockingSync;
		cudaEventCreateWithFlags(&start_event, eventflags);
		cudaEventCreateWithFlags(&stop_event, eventflags);
		cudaEventRecord(start_event, 0);
		for(int i=0; i<CNT; i++) {
			cudaMemcpy(d_recv_data, d_send_data, size*sizeof(float), cudaMemcpyDeviceToDevice);	
		}
		std::cout << "hello, use_cuda_time" << std::endl;
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&time_memcpy, start_event, stop_event);  // ms
		std::cout << "Time is " << time_memcpy/1000. << "s" << std::endl;
		std::cout << "GPU" << id0 << " ---> GPU" << id1 << " :" << 
			WIDTH*HEIGHT*sizeof(float)*CNT*1000./(1024*1024*time_memcpy) << "MB/s" << std::endl;
		cudaEventDestroy(start_event);
		cudaEventDestroy(stop_event);
	} else {
		long long start = getSystemTime();
		for(int i=0; i<CNT; i++) {
			cudaMemcpy(d_recv_data, d_send_data, size*sizeof(float), cudaMemcpyDeviceToDevice);	
			//cudaMemcpyPeer(d_recv_data, id1, d_send_data, id0, size*sizeof(float));	
		}
		std::cout << "debug 1" << std::endl;
		long long end = getSystemTime();
		std::cout << "Time is " << (end-start)/1000. << "s" << std::endl;
		std::cout << "GPU" << id0 << " ---> GPU" << id1 << " :" << 
			WIDTH*HEIGHT*sizeof(float)*CNT*1000./(1024*1024*(end - start+1)) << "MB/s" << std::endl;
	}			//WIDTH*HEIGHT*4.*CNT/(1000*(end - start)) << "Mb/s" << std::endl;
}

int main(int argc, char **argv)
{
	// define the ptr
	int size = WIDTH * HEIGHT;
	float *h_data, *d_send_data, *d_recv_data; 
	bool use_cuda_time = true;

	if(argc != 3) {
		std::cout << "the number of paramter should be equal 2" << std::endl;
		std::cout << "egs: bandwidth_test_between2gpu 0 1" << std::endl;
	}
	//std::cout << "debug 1" << std::endl;
	int id0 = atoi(argv[1]);
	int id1 = atoi(argv[2]);
	std::cout << "id0=" << id0 << ", id1=" << id1 << std::endl;

	//h_data = new float[size];
	cudaMallocHost(&h_data, size*sizeof(float));
	init_data(h_data, size);

	cudaSetDevice(id0);
	cudaMalloc(&d_send_data, size*sizeof(float));
	cudaSetDevice(id1);
	cudaMalloc(&d_recv_data, size*sizeof(float));
	cudaMemcpy(d_send_data, h_data, size*sizeof(float), cudaMemcpyHostToDevice);

	int can_access_peer_0_1, can_access_peer_1_0;
	cudaSetDevice(id0);
	CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer_0_1, id0, id1));
	CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer_1_0, id1, id0));

	if(can_access_peer_0_1 && can_access_peer_1_0) {
		std::cout << "can GPU" << id0 << "access from GPU" << id1 << ": Yes" << std::endl;
		cudaSetDevice(id0);
		CUDA_CHECK(cudaDeviceEnablePeerAccess(id1, 0));
		cudaSetDevice(id1);
		CUDA_CHECK(cudaDeviceEnablePeerAccess(id0, 0));
	} else {
		std::cout << "can GPU" << id0 << "access from GPU" << id1 << ": No" << std::endl;
	}

	cudaSetDevice(id0);
	//use_cuda_time = false;
	use_cuda_time = true;
	test_2gpu(d_send_data, d_recv_data, size, id0, id1, use_cuda_time);

	cudaFreeHost(h_data);
	cudaFree(d_send_data);
	cudaFree(d_recv_data);

	return 0;
}
