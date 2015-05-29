/*
 	this code is just test how to build .cu which deponds on openmp, like using #pragma omp parallel,
	and to learn how to write Makefile
 */

#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>

int main()
{
	omp_set_num_threads(3);
	//#pragma omp parallel num_threads(3)
	#pragma omp parallel
	{
		std::cout << "Hello World!" << std::endl;
	}
	return 0;
}
