#include "Renderer.h"

#include <iostream>
#include <thread>
#include <device_atomic_functions.h>

#include "Geometryrepository.h"
#include "Raytracer.h"

namespace GPURenderer {
	__global__ static void render_init(int width, int height, curandState* state) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		int idx = j + i * height;
		curand_init(19260817, idx, 0,  &state[idx]);
	}

	__global__ static void Dorender(Camera* pcam, la::vec3* pbg, size_t root, int width, int height,
		la::vec3* pixels, int nrays, curandState* states, volatile int* progress) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		auto idx = j + i * height;
		if ((i >= width) || (j >= height)) return;
		auto col = la::vec3(0);
		Cuboid* cub = static_cast<Cuboid*>(Geometryrepository::GetGeo(root));
		la::vec3 bg = *pbg;
		for (auto k = 0; k < nrays; ++k) {
			Ray ray = pcam->GenRay(i, j, width, height, states[idx]);
		//	printf("Dorender: Get ray (%f, %f, %f)->(%f, %f, %f)\n", ray.GetPos().x, ray.GetPos().y, ray.GetPos().z,
			//	ray.GetDir().x, ray.GetDir().y, ray.GetDir().z);
			col += Raytracing::RayTracing(ray, cub, bg, states[idx]);
		}
		pixels[idx] = sqrt(col / (float)nrays); // Gamma correction
		if (!(threadIdx.x || threadIdx.y)) {
			atomicAdd((int*)progress, 1);
			__threadfence_system();
		}
	}

	void Render(World& world, Image3& output, int nrays) {
		world.CreateHierarchy();
		Geometryrepository::Initiate(world);
		Materialrepository::Sendtogpu();
		Geometryrepository::Sendtogpu();

		curandState* curandstates;
		size_t root = world.GetRoot();
		printf("%llu\n", root);
		int const width = output.GetWidth(), height = output.GetHeight();
		int const npixels = width * height;
		la::vec3* pixels;
		Camera* pcam;
		la::vec3* pbg;
		
		printf("Just enter\n");
		checkCudaErrors(cudaMalloc(&curandstates, sizeof(curandState) * npixels));
		checkCudaErrors(cudaMalloc(&pixels, sizeof(la::vec3) * npixels));
		checkCudaErrors(cudaMalloc(&pcam, sizeof(Camera)));
		checkCudaErrors(cudaMemcpy(pcam, &world.GetCam(), sizeof(Camera),
			cudaMemcpyKind::cudaMemcpyHostToDevice));
		la::vec3 background = world.GetBackground();
		checkCudaErrors(cudaMalloc(&pbg, sizeof(la::vec3)));
		checkCudaErrors(cudaMemcpy(pbg, &background, sizeof(la::vec3),
			cudaMemcpyKind::cudaMemcpyHostToDevice));

		printf("Finish memory allocating\n");
		int const tx = 16, ty = 16;
		dim3 const blocks(width / tx + 1, height / ty + 1);
		dim3 const threads(tx, ty);
		render_init <<<blocks, threads >>> (width, height, curandstates);
		checkCudaErrors(cudaDeviceSynchronize());
		printf("Finish randstate init\n");
		checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8192 << 1)); // Add stack size
		volatile int* d_prog = nullptr;
		volatile int* progress = nullptr;
		int nblocks = blocks.x * blocks.y;
		checkCudaErrors(cudaHostAlloc((void**)&progress, sizeof(int), cudaHostAllocMapped));
		checkCudaErrors(cudaHostGetDevicePointer((int**)&d_prog, (int*)progress, 0));
		*progress = 0;
		Dorender <<<blocks, threads>>> (pcam, pbg, root, width, height, pixels, nrays, curandstates, d_prog);
		printf("Progress:\n");
		float lastprogress = 0.0f;
		do {
			int val1 = *progress;
			float kern_progress = (float)val1 / nblocks;
			if (kern_progress - lastprogress > 0.05f) {
				printf("percent complete: %2.2f %%\n", kern_progress * 100);
				lastprogress = kern_progress;
			}
		} while (lastprogress < 0.95f);
		checkCudaErrors(cudaDeviceSynchronize());
		printf("Finish rendering\n");
		auto cpupixels = new la::vec3[npixels];
		checkCudaErrors(cudaMemcpy(cpupixels, pixels, sizeof(la::vec3) * npixels,
			cudaMemcpyKind::cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(pixels));
		for (size_t i = 0; i < npixels; ++i) {
			output.Setcol((int)i / height, (int)i % height, cpupixels[i], true);
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
		threads.push_back(std::thread(&DoRender, std::ref(world), std::ref(output),
			Totpixel * i/ processor_count, Totpixel * (i + 1) / processor_count, nrays,
			OutputFreq, i, std::ref(OutputLock)));
	}

	for (auto& th : threads) {
		th.join();
	}
}

void Renderer::DoRender(World& world, Image3& output, uint32_t from, uint32_t to, int nrays,
	int OutputFreq, int nthread, std::mutex& lock) {
	Camera const& cam = world.GetCam();
	int const Width = output.GetWidth(), Height = output.GetHeight();
	std::vector<Ray> rays;
	std::vector<la::vec3> cols;
	for (auto ind = from; ind < to; ++ind) {
		auto i = ind / Height, j = ind % Height;
		cam.GenRay(i, j, Width, Height, rays, nrays);
		la::vec3 col = la::vec3(0.0f);
		for (Ray const& ray : rays) {
			col += world.RayTracing(ray);
		}
		col = col / (float)nrays;
		col = sqrt(col); // Gamma correction
		cols.push_back(col);
		rays.clear();
		if ((ind - from) % OutputFreq == 0) {
			printf("Thread %d: Complete %f percent.\n", nthread, ((float)ind - from) * 100 / (to - from));
		}
	}
	printf("Thread %d: Finish color collecting. Now writing to the output:\n", nthread);
	{
		std::lock_guard<std::mutex> lk(lock);
		for (size_t ind = from; ind < to; ++ind) {
			output.Setcol((int)ind / Height, (int)ind % Height,
				la::clamp(cols[ind - from], la::vec3(0), la::vec3(1)), true);
		}
	}
	printf("Thread %d: Rendering task completed.\n", nthread);
}
