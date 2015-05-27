#include <mpi.h>
#include <iostream>

//using namespace MPI;

int main()
{
	int rank, size, len;
	char version[MPI_MAX_LIBRARY_VERSION_STRING];

	MPI::Init();

	// version I
	//rank = MPI::COMM_WORLD.Get_rank();
	//size = MPI::COMM_WORLD.Get_size();

	// version II
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	MPI_Get_library_version(version, &len);

	std::cout << "Hello, world!  I am " << rank << " of " << size << "(" << version << ", " << len << ")" << std::endl;
	MPI::Finalize();

	return 0;
}
