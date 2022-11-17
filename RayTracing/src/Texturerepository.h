#pragma once

#include <memory>
#include <vector>
#include "Texture.h"

namespace Texturerepository {
	uint32_t AddTex(std::shared_ptr<Texture> tex);
	void Clearmemory();
	void SendtoGPU();
	__host__ __device__ la::vec3 GetColor(uint32_t texid, uint32_t x, uint32_t y);
	std::vector<std::shared_ptr<Texture>>& GetTexs();
};
