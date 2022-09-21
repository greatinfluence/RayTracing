#include "Testtool.h"

std::ostream& operator<<(std::ostream& out, la::vec3 vec)
{
	out << vec.x << ' ' << vec.y << ' ' << vec.z << std::endl;
	return out;
}

