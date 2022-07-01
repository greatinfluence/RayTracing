#include "Geometryrepository.h"
#include "World.h"
namespace Geometryrepository {
	__device__ Geometry** g_Geos;
	Geometry** g_Geos_cpu;
	std::vector<std::shared_ptr<Geometry>> m_Geos;

	void Initiate(World const& world) {
		size_t size = world.GetNgeo();
		for (size_t i = 0; i < size; ++i) {
			m_Geos.push_back(world.GetGeo(i));
		}
	}

	void Clearmemory() {
	}

	void Sendtogpu() {
	}

	__host__ __device__ Geometry* GetGeo(size_t geoid) {
	#ifdef __CUDA_ARCH__
		// Device
		return g_Geos[geoid];
	#else
		// Host
		return m_Geos[geoid].get();
	#endif
	}

}
