#include "Materialrepository.h"

#include "cuda_runtime.h"
#include <cstdio>

//__device__ Material** Materialrepository::g_Mats = nullptr;
//Material** Materialrepository::g_Mats_cpu = nullptr;
//std::vector<std::shared_ptr<Material>> Materialrepository::m_Mats{};

namespace Materialrepository {
	__device__ Material** g_Mats;
	Material** g_Mats_cpu;
	std::vector<std::shared_ptr<Material>> m_Mats;

	uint32_t AddMat(std::shared_ptr<Material> mat) {
		m_Mats.push_back(mat);
		return (uint32_t)m_Mats.size() - 1;
	}

	__global__ void ClearMaterials(size_t size) {
		int ind = threadIdx.x + blockIdx.x * blockDim.x;
		if(ind < size) delete g_Mats[ind];
	}

	void CleanMemory() {
		if (g_Mats_cpu != nullptr) {
			size_t size = m_Mats.size();
			dim3 tr(32), block(size / 32 + 1);
			ClearMaterials <<<block, tr>>>(size);
			checkCudaErrors(cudaFree(g_Mats_cpu));
		}
	}

	__global__ void CreateDiffuseonGPU(Material*& place, la::vec3 gloom, la::vec3 albedo) {
		place = new Diffuse(gloom, albedo);
	}

	__global__ void CreateMetalonGPU(Material*& place, la::vec3 gloom, la::vec3 albedo, float fuzz) {
		place = new Metal(gloom, albedo, fuzz);
	}

	__global__ void CreateDieletriconGPU(Material*& place, la::vec3 gloom, float ir) {
		place = new Dieletric(gloom, ir);
	}

	void Sendtogpu() {
		if (g_Mats_cpu != nullptr) {
			printf("Sendtogpu Error: The material has already sent to GPU!\n");
		}
		size_t size = m_Mats.size();
		checkCudaErrors(cudaMallocManaged(&g_Mats_cpu, sizeof(Material*) * size));
		for (auto i = 0; i < size; ++i) {
			switch (m_Mats[i]->GetType()) {
			case MatType::Diffuse: {
				auto* dif = static_cast<Diffuse*>(m_Mats[i].get());
				CreateDiffuseonGPU <<<1, 1 >>> (g_Mats_cpu[i], dif->GetGlow(), dif->GetAlbedo());
				break;
			}
			case MatType::Metal: {
				auto* met = static_cast<Metal*>(m_Mats[i].get());
				CreateMetalonGPU <<<1, 1>>> (g_Mats_cpu[i], met->GetGlow(), met->GetAlbedo(), met->GetFuzz());
				break;
			}
			case MatType::Dieletric: {
				auto* dil = static_cast<Dieletric*>(m_Mats[i].get());
				CreateDieletriconGPU <<<1, 1>>> (g_Mats_cpu[i], dil->GetGlow(), dil->GetIr());
				break;
			}
			default: {
				printf("Materialrepository::SendtuGPU error: Unknown Mat type %d\n", (int)m_Mats[i]->GetType());
			}
			}
		}
		checkCudaErrors(cudaMemcpyToSymbol(g_Mats, &g_Mats_cpu, sizeof(Material**), 0,
			cudaMemcpyKind::cudaMemcpyHostToDevice));
	}

	__host__ __device__ Material* GetMat(uint32_t matid) {
	#ifdef __CUDA_ARCH__
		// Device
		return g_Mats[matid];
	#else
		// Host
		return m_Mats[matid].get();
	#endif
	}

	std::vector<std::shared_ptr<Material>>& GetMats() {
		return m_Mats;
	}
}
