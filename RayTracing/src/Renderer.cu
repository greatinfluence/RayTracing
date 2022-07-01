#include "Renderer.h"

#include <iostream>
#include <thread>

#include "Geometryrepository.h"

namespace GPURenderer {
	__global__ static void render_init(int width, int height, curandState* state) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		int idx = j * width + i;
		curand_init(19260817, idx, 0,  &state[idx]);
	}

	__global__ void Dorender(Camera const& cam, int width, int height, glm::vec3* pixels,
		int nrays, curandState* randstates) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		pixels[j * width + i] = glm::vec3(i, j, 0);
	}

	void Render(World& world, Image3& output, int nrays) {
		world.CreateHierarchy();
		Geometryrepository::Initiate(world);
		Materialrepository::Sendtogpu();
		Geometryrepository::Sendtogpu();

		curandState* curandstates;
		int width = output.GetWidth(), height = output.GetHeight();
		int npixels = width * height;
		glm::vec3* pixels;
		
		printf("Just enter\n");
		checkCudaErrors(cudaMalloc(&curandstates, sizeof(curandState) * npixels));
		checkCudaErrors(cudaMallocManaged(&pixels, sizeof(glm::vec3) * npixels));
		printf("Finish memory allocating\n");
		int tx = 8, ty = 8;
		dim3 blocks(width / tx + 1, height / ty + 1);
		dim3 threads(tx, ty);
		render_init <<<blocks, threads >>> (width, height, curandstates);
		printf("Finish randstate init\n");
		Dorender<<<blocks, threads>>>(world.GetCam(), output.GetWidth(), output.GetHeight(),
			pixels, nrays, curandstates);
		printf("Finish rendering\n");
		glm::vec3* cpupixels = new glm::vec3[npixels];
		checkCudaErrors(cudaMemcpy(cpupixels, pixels, sizeof(glm::vec3) * npixels, cudaMemcpyKind::cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < npixels; ++i) {
			output.Setcol(i / height, i % height, cpupixels[i], true);
		}
		printf("Finish everything\n");
		Materialrepository::CleanMemory();
		Geometryrepository::Clearmemory();
		delete[] cpupixels;
	}


}

void Renderer::Render(World& world, Image3& output, int nrays, int OutputFreq) {
	world.CreateHierarchy();
	Geometryrepository::Initiate(world);
	
	std::vector<std::thread> threads;
	std::mutex OutputLock;

	auto const processor_count = std::thread::hardware_concurrency();
	int const Totpixel = output.GetWidth() * output.GetHeight();
	for (uint32_t i = 0; i < processor_count; ++i) {
		threads.push_back(std::thread(&DoRender, std::ref(world), std::ref(output), Totpixel * i/ processor_count,
			Totpixel * (i + 1) / processor_count, nrays, OutputFreq, i, std::ref(OutputLock)));
	}

	for (auto& th : threads) {
		th.join();
	}
}

void Renderer::DoRender(World& world, Image3& output, uint32_t from, uint32_t to, int nrays, int OutputFreq, int nthread, std::mutex& lock) {
	Camera const& cam = world.GetCam();
	int const Width = output.GetWidth(), Height = output.GetHeight();
	std::vector<Ray> rays;
	std::vector<glm::vec3> cols;
	for (auto ind = from; ind < to; ++ind) {
		auto i = ind / Height, j = ind % Height;
		cam.GenRay(i, j, Width, Height, rays, nrays);
		auto col = glm::vec3(0.0f);
		for (Ray const& ray : rays) {
			col += world.RayTracing(ray);
		}
		col = col / (float)nrays;
		col = sqrt(col); // Gamma correction
		cols.push_back(col);
		rays.clear();
		if ((ind - from) % OutputFreq == 0) {
			printf("Thread %d: Complete %f %\n", nthread, ((float)ind - from) * 100 / (to - from));
		}
	}
	printf("Thread %d: Finish color collecting. Now writing to the output:\n", nthread);
	{
		std::lock_guard<std::mutex> lk(lock);
		for (size_t ind = from; ind < to; ++ind) {
			output.Setcol(ind / Height, ind % Height, glm::clamp(cols[ind - from], glm::vec3(0), glm::vec3(1)), true);
		}
	}
	printf("Thread %d: Rendering task completed.\n", nthread);
}
