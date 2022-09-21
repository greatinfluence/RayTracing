#pragma once

#include "Ray.h"

class Camera {
public:
	__host__ __device__ Camera()
		:m_Pos{ la::vec3(0, 0, 0) }, m_Front{ la::vec3(1, 0, 0) },
		m_Up{ la::vec3(0, 1, 0) }, m_Horang{ 1.57f }, m_Perang{ 1.57f } {}
	Camera(la::vec3 pos, la::vec3 front, la::vec3 up,
		float horang = 1.57, float perang = 1.57);

	// Genray(x, y, width, height, rays, num) generates num rays on the positioned direction,
	//     place all the generated rays into the vector rays
	void GenRay(int x, int y, int width, int height, std::vector<Ray>& rays, int num = 4) const;

	// Genray(x, y, width, height, state) generates a ray on the positioned direction
	__device__ Ray GenRay(int x, int y, int width, int height, curandState& state) const;

	la::vec3 GetPos() const { return m_Pos; }
	la::vec3 GetFront() const { return m_Front; }
	la::vec3 GetUp() const { return m_Up; }
	float GetHor() const { return m_Horang; }
	float GetPerp() const { return m_Perang; }
private:
	la::vec3 m_Pos;
	la::vec3 m_Front, m_Up;
	// horang denotes the sight angle of the horizontal direction
	//   perang denotes the sight angle of the perpendicular direction
	float m_Horang, m_Perang;
};
