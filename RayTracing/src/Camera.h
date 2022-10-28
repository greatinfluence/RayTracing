#pragma once

#include "Ray.h"

enum class CamType { Unknown = 0, Reg = 1, Fish = 2 };

class Camera {
public:
	__host__ __device__ Camera() {}

	// Genray(x, y, width, height, rays, num) generates num rays on the positioned direction,
	//     place all the generated rays into the vector rays
	virtual void GenRay(int x, int y, int width, int height, std::vector<Ray>& rays, int num = 4) const = 0;

	// Genray(x, y, width, height, state) generates a ray on the positioned direction
	__device__ virtual Ray GenRay(int x, int y, int width, int height, curandState& state) const = 0;
	__host__ __device__ virtual CamType GetType() const = 0;
};

class RegularCamera : public Camera {
public:
	__host__ __device__ RegularCamera() :
		m_Pos{ la::vec3(0) }, m_Front { la::vec3(1.0f, 0, 0) }, m_Up{la::vec3(0, 1.0f, 0)},
		m_Hor{ 2.0f * 2 }, m_Per{ 2.0f } {}

	__host__ __device__ RegularCamera(la::vec3 pos, la::vec3 front, la::vec3 up, double hor = 4.0, double per = 2.0):
		m_Pos{ pos }, m_Front{ front }, m_Up{ up }, m_Hor{ hor }, m_Per{ per } {}

	__host__ __device__ CamType GetType() const override { return CamType::Reg; }

	void GenRay(int x, int y, int width, int height, std::vector<Ray>& rays, int num = 4) const override;

	__device__ Ray GenRay(int x, int y, int width, int height, curandState& state) const override;

	la::vec3 m_Pos, m_Front, m_Up;
	double m_Hor, m_Per;
};

class FishEyeCamera: public Camera {
public:
	__host__ __device__ FishEyeCamera()
		:m_Pos{ la::vec3(0, 0, 0) }, m_Front{ la::vec3(1, 0, 0) },
		m_Up{ la::vec3(0, 1, 0) }, m_Horang{ 1.57f }, m_Perang{ 1.57f } {}
	__host__ __device__ FishEyeCamera(la::vec3 pos, la::vec3 front, la::vec3 up,
		double horang = 1.57, double perang = 1.57): m_Pos(pos), m_Front(la::normalize(front)),
		m_Up(la::normalize(up)), m_Horang(horang), m_Perang(perang) {
	if (abs(la::dot(front, up)) > eps) {
		printf("Error: The FishEyeCamera's up/front vectors are not perpendicular\n");
	}
}

	__host__ __device__ CamType GetType() const override { return CamType::Fish; }

	void GenRay(int x, int y, int width, int height, std::vector<Ray>& rays, int num = 4) const override;

	__device__ Ray GenRay(int x, int y, int width, int height, curandState& state) const override;

	la::vec3 m_Pos, m_Front, m_Up;

	// horang denotes the sight angle of the horizontal direction
	//   perang denotes the sight angle of the perpendicular direction
	double m_Horang, m_Perang;
};
