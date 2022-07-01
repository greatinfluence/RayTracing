#pragma once

#include <vector>
#include <memory>

#include "device_launch_parameters.h"

#include "Material.h"

namespace Materialrepository {
	uint32_t AddMat(std::shared_ptr<Material> mat);
	void CleanMemory();
	void Sendtogpu();
	__host__ __device__ Material* GetMat(uint32_t matid);
	std::vector<std::shared_ptr<Material>>& GetMats();
}
