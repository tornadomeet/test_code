#include <iostream>
#include <omp.h>

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
