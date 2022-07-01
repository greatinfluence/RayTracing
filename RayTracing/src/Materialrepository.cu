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
}

uint32_t Materialrepository::AddMat(std::shared_ptr<Material> mat) {
	m_Mats.push_back(mat);
	return (uint32_t)m_Mats.size() - 1;
}

void Materialrepository::CleanMemory() {
	size_t size = m_Mats.size();
	if (g_Mats_cpu != nullptr) {
		for (size_t i = 0; i < size; ++i) {
			checkCudaErrors(cudaFree(g_Mats_cpu[i]));
		}
		checkCudaErrors(cudaFree(g_Mats_cpu));
	}
}

void Materialrepository::Sendtogpu() {
	if (g_Mats_cpu != nullptr) {
		printf("Sendtogpu Error: The material has already sent to GPU!\n");
	}
	size_t size = m_Mats.size();
	cudaMallocManaged(&g_Mats_cpu, sizeof(Material*) * size);
	for (auto i = 0; i < size; ++i) {
		cudaMalloc(&(g_Mats_cpu[i]), sizeof(*m_Mats[i]));
		cudaMemcpy(g_Mats_cpu[i], m_Mats[i].get(), m_Mats[i] -> GetSize(),
			cudaMemcpyKind::cudaMemcpyHostToDevice);
	}
	cudaMemcpy(&g_Mats, &g_Mats_cpu, sizeof(Material**), cudaMemcpyKind::cudaMemcpyHostToDevice);
}

__host__ __device__ Material* Materialrepository::GetMat(uint32_t matid) {
#ifdef __CUDA_ARCH__
	// Device
	return g_Mats[matid];
#else
	// Host
	return m_Mats[matid].get();
#endif
}

std::vector<std::shared_ptr<Material>>& Materialrepository::GetMats() {
	return m_Mats;
}
