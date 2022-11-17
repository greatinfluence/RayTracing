#include "Texturerepository.h"

#include "cuda_runtime.h"
#include "Settings.h"
#include <cstdio>

namespace Texturerepository {

	struct TexinGPU {
		uint32_t m_Width, m_Height;
		unsigned char* m_Data;
	};

	__device__ TexinGPU* g_Bits;
	TexinGPU* g_Bits_cpu;
	std::vector<std::shared_ptr<Texture>> m_Texs;
	uint32_t AddTex(std::shared_ptr<Texture> tex)
	{
		m_Texs.push_back(tex);
		return uint32_t(m_Texs.size() - 1);
	}

	__global__ void ClearTextures(size_t size) {
		int ind = threadIdx.x + blockIdx.x * blockDim.x;
		if (ind < size) delete g_Bits[ind].m_Data;
	}

	void Clearmemory() {
		auto size = m_Texs.size();
		dim3 tr(32), block(size / 32 + 1);
		ClearTextures <<<block, tr>>>(size);
		checkCudaErrors(cudaFree(g_Bits_cpu));
	}

	__device__ unsigned char* dataptr;

	__global__ void CreateTexonGPU(TexinGPU& place, unsigned int width, unsigned int height) {
		place = TexinGPU{ width, height, dataptr };
	}

	void SendtoGPU() {
		size_t size = m_Texs.size();
		checkCudaErrors(cudaMallocManaged(&g_Bits_cpu, sizeof(TexinGPU) * size));
		for (size_t i = 0; i < size; ++i) {
			auto const& img = m_Texs[i]->m_Img;
			auto sz = (size_t)img.GetWidth() * img.GetHeight() * img.GetChannels();
			checkCudaErrors(cudaMemcpy(dataptr, img.GetData(), sizeof(unsigned char) * sz,
				cudaMemcpyKind::cudaMemcpyHostToDevice));
			CreateTexonGPU <<<1, 1 >>> (g_Bits_cpu[i], img.GetWidth(), img.GetHeight());
		}
		checkCudaErrors(cudaMemcpyToSymbol(g_Bits, &g_Bits_cpu, sizeof(TexinGPU*), 0,
			cudaMemcpyKind::cudaMemcpyHostToDevice));
	}

	std::vector<std::shared_ptr<Texture>>& GetTexs() { return m_Texs; }
}
