#include "Cuboid.h"

#include <limits>
#include <cassert>

#include "la.h"
#include "Ball.h"
#include "Triangle.h"
#include "Ray.h"
#include "Settings.h"
#include "Geometryrepository.h"
#include "World.h"

__host__ __device__ Cuboid::Cuboid()
	: m_Min{ floatmax, floatmax, floatmax }, m_Max{ floatmin, floatmin, floatmin },
	m_Nsubgeo{ 0 }, m_Subgeoid{nullptr} {}

void Cuboid::AppendSubGeos(World const& world, std::vector<size_t> const& subgeos) {
	m_Subgeoid = new size_t[subgeos.size()];
	for (size_t subgeoid : subgeos) {
		auto subgeo = world.GetGeo(subgeoid);
		switch (subgeo->GetType()) {
		case GeoType::Cuboid: {
			auto* subcub = static_cast<Cuboid*>(subgeo.get());
			for (auto i = 0; i < 3; ++i) {
				m_Min[i] = fmin(m_Min[i], subcub->m_Min[i]);
				m_Max[i] = fmax(m_Max[i], subcub->m_Max[i]);
			}
			break;
		}
		case GeoType::Ball: {
			auto* subba = static_cast<Ball*>(subgeo.get());
			la::vec3 cent = subba->m_Center;
			float r = subba->m_Radius;
			for (auto i = 0; i < 3; ++i) {
				m_Min[i] = fmin(m_Min[i], cent[i] - r);
				m_Max[i] = fmax(m_Max[i], cent[i] + r);
			}
			break;
		}
		case GeoType::Triangle: {
			auto* subtr = static_cast<Triangle*>(subgeo.get());
			for (auto i = 0; i < 3; ++i) {
				la::vec3 vert = subtr->GetPos(i);
				for (auto j = 0; j < 3; ++j) {
					m_Min[j] = fmin(m_Min[j], vert[j]);
					m_Max[j] = fmax(m_Max[j], vert[j]);
				}
			}
			break;
		}
		default: {
			printf("Unrecongnized Geometry Type!\n");
			return;
		}
		}
		m_Subgeoid[m_Nsubgeo ++] = subgeoid;
	}
}

__host__ __device__ la::vec3 Cuboid::GetNorm(la::vec3 pos) const
{
	printf("You should never call this function, since it's just a structural function\n");
	return la::vec3(0.0f);
}

//__device__ static void TestRay(Ray ray) {
//	assert(ray.Hit(nullptr) == -1);
//}


void Cuboid::TestHit(Ray const& ray, float& dist, Geometry*& hitted) const
{
//	printf("The memory address of the Cuboid: %p\n", this);
//	printf("The type of Cuboid: %d\n", (int)GeoType::Cuboid);
//	printf("The type of this Cuboid: %d\n", (int)GetType());
//	printf("The type of this Cuboid: %d\n", (int)this->GetType());
//#ifdef __CUDA_ARCH__
//	TestRay(ray);
//#endif
	//printf("Cuboid: Receives the ray: (%f, %f, %f) -> (%f, %f, %f)\n", r.GetPos().x, r.GetPos().y, r.GetPos().z, r.GetDir().x, r.GetDir().y, r.GetDir().z);
//	if (ray.Hit(nullptr) != -1) {
//		printf("Bad hit test\n");
//	}
	if (ray.Hit(this) < dist) {
		// It is possible for the ray to hit the box
//		printf("Cuboid: Has hit the ray\n");
		for(size_t i = 0; i < m_Nsubgeo; ++ i) {
			auto geo = Geometryrepository::GetGeo(m_Subgeoid[i]);
			if (geo->GetType() == GeoType::Cuboid) {
				// A sublevel of Cuboid
				auto* cub = static_cast<Cuboid*>(geo);
				cub->TestHit(ray, dist, hitted);
			}
			else {
				// A common object
				float ndist = ray.Hit(geo);
				if (ndist < dist) {
					dist = ndist;
					hitted = geo;
				}
			}
		}
//		printf("Cuboid: End hit refreshing\n");
	}
//	printf("Cuboid: End ray testing\n");
}

__device__ void Cuboid::TestHitdevice(Ray const& ray, float& dist, Geometry*& hitted) const
{
	//printf("The memory address of the Cuboid: %p\n", this);
	//printf("The type of Cuboid: %d\n", (int)GeoType::Cuboid);
	//printf("The type of this Cuboid: %d\n", (int)GetType());
	//printf("The type of this Cuboid: %d\n", (int)this->GetType());
	//printf("Min: %f %f %f\n", m_Min[0], m_Min[1], m_Min[2]);
	//printf("Max: %f %f %f\n", m_Max[0], m_Max[1], m_Max[2]);
	//printf("Cuboid: Receives the ray: (%f, %f, %f) -> (%f, %f, %f)\n", m_Pos.x, m_Pos.y, m_Pos.z, m_Dir.x, m_Dir.y, m_Dir.z);
	if (ray.Hit(this) < dist) {
		// It is possible for the ray to hit the box
		//printf("Cuboid: Has hit the ray\n");
		for (size_t i = 0; i < m_Nsubgeo; ++ i) {
			auto geo = Geometryrepository::GetGeo(m_Subgeoid[i]);
			if (geo->GetType() == GeoType::Cuboid) {
				// A sublevel of Cuboid
				auto* cub = static_cast<Cuboid*>(geo);
				cub->TestHitdevice(ray, dist, hitted);
			}
			else {
				// A common object
				float ndist = ray.Hit(geo);
				if (ndist < dist) {
					dist = ndist;
					hitted = geo;
				}
			}
		}
		//printf("Cuboid: End hit refreshing\n");
	}
	//printf("Cuboid: End ray testing\n");
}
