#pragma once

#include <iostream>
#include "glm/vec3.hpp"

#ifndef VEC3_OUT
#define VEC3_OUT

std::ostream& operator<<(std::ostream& out, glm::vec3 vec) {
	out << vec.x << ' ' << vec.y << ' ' << vec.z << std::endl;
	return out;
}

#endif

