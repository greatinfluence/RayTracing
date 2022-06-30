#include "Materialrepository.cuh"

#include "cuda_runtime.h"
#include <cstdio>

uint32_t Materialrepository::AddMat(std::shared_ptr<Material> mat) {
	if (Matrep::g_Mats != nullptr) {
		printf("Possible error: The materials are already sent to GPU!\n");
	}
	Matrep::m_Mats.push_back(mat);
	return Matrep::m_Mats.size() - 1;
}

Material** Materialrepository::Sendtogpu() {
	uint64_t nmats = Matrep::m_Mats.size();
	cudaMallocManaged(&Matrep::g_Mats, sizeof(Material*) * nmats);
	for (auto i = 0; i < nmats; ++i) {
		cudaMalloc(&(Matrep::g_Mats[i]), sizeof(*Matrep::m_Mats[i]));
		cudaMemcpy(Matrep::g_Mats[i], Matrep::m_Mats[i].get(), sizeof(*Matrep::m_Mats[i]),
			cudaMemcpyKind::cudaMemcpyHostToDevice);
	}
	return Matrep::g_Mats;
}

__host__ __device__ Material* Materialrepository::GetMat(uint32_t matid) {
#ifdef __CUDA_ARCH__
	// Device
	return Matrep::g_Mats[matid];

#else
	// Host
	return Matrep::m_Mats[matid].get();
#endif
}

Materialrepository::~Materialrepository() {
	if (Matrep::g_Mats != nullptr) {
		uint64_t nmats = Matrep::m_Mats.size();
		for (auto i = 0; i < nmats; ++i) {
			cudaFree(Matrep::g_Mats[i]);
			cudaFree(Matrep::g_Mats);
		}
	}
}
