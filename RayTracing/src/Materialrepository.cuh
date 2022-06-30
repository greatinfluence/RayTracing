#pragma once

#include <vector>
#include <memory>

#include "device_launch_parameters.h"

#include "Material.h"

namespace Matrep {
	__device__ static Material** g_Mats;
	static std::vector<std::shared_ptr<Material>> m_Mats;
}

class Materialrepository {
public:
	~Materialrepository();
	static uint32_t AddMat(std::shared_ptr<Material> mat);
	static Material** Sendtogpu();
	__host__ __device__ static Material* GetMat(uint32_t matid);
};
