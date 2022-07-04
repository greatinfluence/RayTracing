#include "Geometryrepository.h"
#include "World.h"
namespace Geometryrepository {
	__device__ Geometry** g_Geos;
	Geometry** g_Geos_cpu;
	std::vector<std::shared_ptr<Geometry>> m_Geos;
	std::vector<size_t*> m_Subs;

	void Initiate(World const& world) {
		size_t size = world.GetNgeo();
		for (size_t i = 0; i < size; ++i) {
			m_Geos.push_back(world.GetGeo(i));
		}
	}

	__global__ void ClearGeometries(size_t size) {
		int ind = threadIdx.x + blockIdx.x * blockDim.x;
		if(ind < size) delete g_Geos[ind];
	}

	void Clearmemory() {
		size_t size = m_Geos.size();
		for (auto& sub : m_Subs) {
			checkCudaErrors(cudaFree(sub));
		}
		if (g_Geos_cpu != nullptr) {
			dim3 tr(32), blk(size/32 + 1);
			ClearGeometries<<<blk, tr>>>(size);
			checkCudaErrors(cudaFree(g_Geos_cpu));
		}
	}

	__global__ void CreateCubonGPU(Geometry*& place, glm::vec3 mins, glm::vec3 maxs,
		size_t nsubgeo, size_t* subgeos, size_t matid) {
		auto cub = new Cuboid;
		for (auto i = 0; i < 3; ++i) {
			cub->SetMin(i, mins[i]);
			cub->SetMax(i, maxs[i]);
		}
		cub->SetNsubgeo(nsubgeo);
		cub->SetSubgeoid(subgeos);
		cub->AddMaterial(matid);
		place = cub;
	}

	__global__ void CreateBallonGPU(Geometry*& place, glm::vec3 cent, float rad, size_t matid) {
		auto ball = new Ball(cent, rad);
		ball->AddMaterial(matid);
		place = ball;
	}

	__global__ void CreateTriangleonGPU(Geometry*& place, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
		glm::vec3 norm, size_t matid) {
		auto tri = new Triangle(v1, v2, v3, norm);
		tri->AddMaterial(matid);
		place = tri;
	}

	void Sendtogpu() {
		if (g_Geos_cpu != nullptr) {
			printf("Sendtogpu Error: The geometry has already sent to GPU!\n");
		}
		size_t size = m_Geos.size();
		checkCudaErrors(cudaMallocManaged(&g_Geos_cpu, sizeof(Material*) * size));
		for (auto i = 0; i < size; ++i) {
			switch (m_Geos[i]->GetType()) {
			case GeoType::Cuboid: {
				auto cub = static_cast<Cuboid*>(m_Geos[i].get());
				size_t* ptr = nullptr;
				checkCudaErrors(cudaMalloc(&ptr, sizeof(size_t) * (cub->GetNsubgeo())));
				checkCudaErrors(cudaMemcpy(ptr, cub->GetSubgeoids(), sizeof(size_t) * (cub->GetNsubgeo()),
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				m_Subs.push_back(ptr);
				CreateCubonGPU<<<1, 1>>>(g_Geos_cpu[i],
					glm::vec3(cub->GetMin(0), cub->GetMin(1), cub->GetMin(2)),
					glm::vec3(cub->GetMax(0), cub->GetMax(1), cub->GetMax(2)),
					cub->GetNsubgeo(), ptr, cub->GetMatid());
				break;
			}
			case GeoType::Ball: {
				auto ball = static_cast<Ball*>(m_Geos[i].get());
				CreateBallonGPU<<<1, 1>>>(g_Geos_cpu[i], ball->GetCenter(), ball->GetRadius(), ball->GetMatid());
				break;
			}
			case GeoType::Triangle: {
				auto tri = static_cast<Triangle*>(m_Geos[i].get());
				CreateTriangleonGPU<<<1, 1>>>(g_Geos_cpu[i], tri->GetPos(0), tri->GetPos(1), tri->GetPos(2),
					tri->GetNorm(glm::vec3(0)), tri->GetMatid());
				break;
			}
			default: {
				printf("Geometryrepository::SendToGPU error: Unrecognized geotype %d\n", (int)m_Geos[i]->GetType());
			}
			}
		}
		checkCudaErrors(cudaMemcpyToSymbol(g_Geos, &g_Geos_cpu, sizeof(Material**), 0,
			cudaMemcpyKind::cudaMemcpyHostToDevice));
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
