#pragma once

#include <iostream>
#include "glm/vec3.hpp"

std::ostream& operator<<(std::ostream& out, glm::vec3 vec) {
	out << vec.x << ' ' << vec.y << ' ' << vec.z << std::endl;
	return out;
}


