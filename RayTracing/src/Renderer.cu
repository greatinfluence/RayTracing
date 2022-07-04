#include "Renderer.h"

#include <iostream>
#include <thread>

#include "Geometryrepository.h"
#include "Raytracer.h"

namespace GPURenderer {
	__global__ static void render_init(int width, int height, curandState* state) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= width) || (j >= height)) return;
		int idx = j * width + i;
		curand_init(19260817, idx, 0,  &state[idx]);
	}

	__global__ void Dorender(Camera* pcam, glm::vec3* pbg, size_t* root, int* width, int* height,
		glm::vec3* pixels, int nrays, curandState* states) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		auto idx = j * *width + i;
		if ((i >= *width) || (j >= *height)) return;
		auto col = glm::vec3(0);
		for (auto k = 0; k < nrays; ++k) {
			Ray ray = pcam->GenRay(i, j, *width, *height, states[idx]);
			printf("Dorender: Get ray (%f, %f, %f)->(%f, %f, %f)\n", ray.GetPos().x, ray.GetPos().y, ray.GetPos().z,
				ray.GetDir().x, ray.GetDir().y, ray.GetDir().z);
			col += Raytracing::RayTracing(ray, root, *pbg, states[idx]);
		}
		pixels[idx] = col / (float)nrays;
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
		glm::vec3* pixels;
		Camera* pcam;
		glm::vec3* pbg;
		
		printf("Just enter\n");
		checkCudaErrors(cudaMalloc(&curandstates, sizeof(curandState) * npixels));
		checkCudaErrors(cudaMalloc(&pixels, sizeof(glm::vec3) * npixels));
		checkCudaErrors(cudaMalloc(&pcam, sizeof(Camera)));
		checkCudaErrors(cudaMemcpy(pcam, &world.GetCam(), sizeof(Camera),
			cudaMemcpyKind::cudaMemcpyHostToDevice));
		size_t* proot = nullptr;
		int* pwidth = nullptr;
		int* pheight = nullptr;
		checkCudaErrors(cudaMalloc(&proot, sizeof(size_t)));
		checkCudaErrors(cudaMemcpy(proot, &root, sizeof(size_t),
			cudaMemcpyKind::cudaMemcpyHostToDevice));
		glm::vec3 background = world.GetBackground();
		checkCudaErrors(cudaMalloc(&pbg, sizeof(glm::vec3)));
		checkCudaErrors(cudaMemcpy(pbg, &background, sizeof(glm::vec3),
			cudaMemcpyKind::cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc(&pwidth, sizeof(int)));
		checkCudaErrors(cudaMemcpy(pwidth, &width, sizeof(int),
			cudaMemcpyKind::cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc(&pheight, sizeof(int)));
		checkCudaErrors(cudaMemcpy(pheight, &height, sizeof(int),
			cudaMemcpyKind::cudaMemcpyHostToDevice));

		printf("Finish memory allocating\n");
		int const tx = 8, ty = 8;
		dim3 const blocks(width / tx + 1, height / ty + 1);
		dim3 const threads(tx, ty);
		render_init <<<blocks, threads >>> (width, height, curandstates);
		checkCudaErrors(cudaDeviceSynchronize());
		printf("Finish randstate init\n");
		Dorender <<<blocks, threads>>> (pcam, pbg, proot, pwidth, pheight, pixels, nrays, curandstates);
		checkCudaErrors(cudaDeviceSynchronize());
		printf("Finish rendering\n");
		return;
		auto cpupixels = new glm::vec3[npixels];
		checkCudaErrors(cudaMemcpy(cpupixels, pixels, sizeof(glm::vec3) * npixels,
			cudaMemcpyKind::cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(pixels));
		for (size_t i = 0; i < npixels; ++i) {
			output.Setcol((int)i / height, (int)i % height, cpupixels[i], true);
		}
		printf("Finish everything\n");
		Materialrepository::CleanMemory();
		Geometryrepository::Clearmemory();
		checkCudaErrors(cudaFree(pwidth));
		checkCudaErrors(cudaFree(pheight));
		checkCudaErrors(cudaFree(proot));
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
			printf("Thread %d: Complete %f percent.\n", nthread, ((float)ind - from) * 100 / (to - from));
		}
	}
	printf("Thread %d: Finish color collecting. Now writing to the output:\n", nthread);
	{
		std::lock_guard<std::mutex> lk(lock);
		for (size_t ind = from; ind < to; ++ind) {
			output.Setcol((int)ind / Height, (int)ind % Height,
				glm::clamp(cols[ind - from], glm::vec3(0), glm::vec3(1)), true);
		}
	}
	printf("Thread %d: Rendering task completed.\n", nthread);
}
