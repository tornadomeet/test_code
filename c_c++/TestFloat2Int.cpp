nclude "stdafx.h"
#include <iostream>


int main(int argc, _TCHAR* argv[])
{
	float fvalue = -5.6;
	int ivalue = fvalue;  // 只取整数部分
	std::cout << fvalue << ": " << ivalue << std::endl;

	return 0;
}
