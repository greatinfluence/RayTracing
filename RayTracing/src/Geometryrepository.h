#pragma once

#include "Geometry.h"
#include "Ball.h"
#include "Triangle.h"
#include "Cuboid.h"
#include "Cylindsurf.h"
#include "Plate.h"

class World;

namespace Geometryrepository {
	void Initiate(World const& world);
	void Clearmemory();
	void Sendtogpu();
	__host__ __device__ Geometry* GetGeo(size_t geoid);
}
