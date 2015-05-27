#include <mpi.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include<sys/timeb.h>
#include <cuda_runtime.h>

#define WIDTH 10000
#define HEIGHT 10000
#define CNT 20
// Error handling macros
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
	std::cerr << "MPI error calling \""#call"\"\n"; \
	my_abort(-1); }

// Shut down MPI cleanly if something goes wrong
void my_abort(int err)
{
	std::cout << "Test FAILED\n";
	MPI_Abort(MPI_COMM_WORLD, err);
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

void test_d2d_gpudirect(int rank, float *d_send_data, float *d_recv_data, int size)
{
	MPI_Status status;
	long long start = getSystemTime();
	for(int i=0; i<CNT; i++) {
		//MPI::Comm::Send(d_send_data, size, MPI_FLOAT, 1, 4179);
		//MPI::Comm::Recv(d_recv_data, size, MPI_FLOAT, 0, 4179);
		if(rank == 0) {
			MPI_Send(d_send_data, size, MPI_FLOAT, 1, i, MPI_COMM_WORLD);
		} else if (rank == 1) {
			MPI_Recv(d_recv_data, size, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
		}
	}
	long long end = getSystemTime();
	if(rank == 0) {
		std::cout << "d2d_gpudirect: host0(gpu0) --> host1(gpu0) " << 
			WIDTH*HEIGHT*4.*CNT*1000/(1024*1024*(end - start)) << "Mb/s" << std::endl;
	}
}

void test_d2d_no_gpudirect(int rank, float *d_send_data, float *h_send_data, float *d_recv_data, float *h_recv_data, int size)
{
	MPI_Status status;
	long long start = getSystemTime();
	for(int i=0; i<CNT; i++) {
		//MPI::Comm::Send(d_send_data, size, MPI_FLOAT, 1, 4179);
		//MPI::Comm::Recv(d_recv_data, size, MPI_FLOAT, 0, 4179);
		if(rank == 0) {
			cudaMemcpy(h_send_data, d_send_data, size*sizeof(float), cudaMemcpyDeviceToHost);
			MPI_Send(h_send_data, size, MPI_FLOAT, 1, i, MPI_COMM_WORLD);
		} else if (rank == 1) {
			MPI_Recv(h_recv_data, size, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
			cudaMemcpy(d_recv_data, h_recv_data, size*sizeof(float), cudaMemcpyHostToDevice);
		}
	}
	long long end = getSystemTime();
	if(rank == 0) {
		std::cout << "d2d_no_gpudirect: host0(gpu0) --> host1(gpu0) " << 
			WIDTH*HEIGHT*4.*CNT*1000/(1024*1024*(end - start)) << "Mb/s" << std::endl;
	}
}

void test_h2h(int rank, float *h_send_data, float *h_recv_data, int size)
{
	MPI_Status status;
	long long start = getSystemTime();
	for(int i=0; i<CNT; i++) {
		//MPI::Comm::Send(d_send_data, size, MPI_FLOAT, 1, 4179);
		//MPI::Comm::Recv(d_recv_data, size, MPI_FLOAT, 0, 4179);
		if(rank == 0) {
			MPI_Send(h_send_data, size, MPI_FLOAT, 1, i, MPI_COMM_WORLD);
		} else if (rank == 1) {
			MPI_Recv(h_recv_data, size, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
		}
	}
	long long end = getSystemTime();
	if(rank == 0) {
		std::cout << "h2h: host0(cpu0) --> host1(cpu0) " << 
			WIDTH*HEIGHT*4.*CNT*1000/(1024*1024*(end - start)) << "Mb/s" << std::endl;
	}
}

int main()
{
	int rank, size;
	char name[MPI_MAX_PROCESSOR_NAME];
	int name_len = 0;

	MPI::Init();

	// version II
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
	MPI::Get_processor_name(name, name_len);
	std::cout << rank << " of " << size << "(" << std::string(name) << ")" << std::endl;
	if(size != 2) {
		std::cout << "number of rank is large than 2, so quit" << std::endl;
	}

	// define the ptr
	int size_data = WIDTH * HEIGHT;
	float *h_send_data, *d_send_data, *h_recv_data, *d_recv_data; 
	cudaMallocHost(&h_send_data, size_data*sizeof(float));
	cudaMallocHost(&h_recv_data, size_data*sizeof(float));
	cudaMalloc(&d_send_data, size_data*sizeof(float));
	cudaMalloc(&d_recv_data, size_data*sizeof(float));
	if(rank == 0) {
		init_data(h_send_data, size_data);
		std::cout << "in rank 0, done init_data()" << std::endl;
	}

	test_d2d_gpudirect(rank, d_send_data, d_recv_data, size_data);
	test_d2d_no_gpudirect(rank, d_send_data, h_send_data, d_recv_data, h_recv_data, size_data);
	test_h2h(rank, h_send_data, h_recv_data, size_data);

	cudaFreeHost(h_send_data);
	cudaFreeHost(h_recv_data);
	cudaFree(d_send_data);
	cudaFree(d_recv_data);
	MPI::Finalize();

	return 0;
}
