#pragma once

#include "Ray.h"

class Camera {
public:
	Camera() = delete;
	Camera(glm::vec3 pos, glm::vec3 front, glm::vec3 up,
		float horang = 1.57, float perang = 1.57);
	void GenRay(int x, int y, int width, int height, std::vector<Ray>& rays, int num = 4);
private:
	glm::vec3 m_Pos;
	glm::vec3 m_Front, m_Up;
	// horang denotes the sight angle of the horizontal direction
	//   perang denotes the sight angle of the perpendicular direction
	float m_Horang, m_Perang;
};
