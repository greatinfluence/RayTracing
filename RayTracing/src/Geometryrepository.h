#pragma once

#include "Geometry.h"

class World;

namespace Geometryrepository {
	void Initiate(World const& world);
	void Clearmemory();
	void Sendtogpu();
	__host__ __device__ Geometry* GetGeo(uint32_t geoid);
}
