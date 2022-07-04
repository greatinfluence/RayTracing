#include "Raytracer.h"

#include "Geometryrepository.h"

__device__ glm::vec3 Raytracing::RayTracing(Ray ray, size_t* root, glm::vec3 m_Background,
	curandState& state, int lev, glm::vec3 coef) {
	float dist = floatmax;
	Geometry* hitted = nullptr;
	if (fabs(ray.GetDir().x) > 1.5f) {
		printf("What?\n");
	}
	printf("Raytracing: get ray (%f, %f, %f) -> (%f, %f, %f)\n", ray.GetPos().x, ray.GetPos().y, ray.GetPos().z,
		ray.GetDir().x, ray.GetDir().y, ray.GetDir().z);
	printf("Begin hit check\n");
	static_cast<Cuboid*>(Geometryrepository::GetGeo(*root))->TestHit(ray, dist, hitted);
	printf("End hit check\n");
	if (hitted != nullptr) {
		glm::vec3 hitpos = ray.GetPos() + ray.GetDir() * dist;
		glm::vec3 att, wi, norm = hitted->GetNorm(hitpos);
		Material* mat = hitted->GetMaterial();
		float poss = mat->scatter(hitpos, -ray.GetDir(),
			norm, att, wi, &state);
		if (fabs(dist) > 1e3 || fabs(ray.GetDir().x) > 1.5f) {
			// Impossible!
			printf("What?\n");
		}
		if(lev > 4) {
			float rr = glm::clamp(fmax(coef.r, fmax(coef.g, coef.b)), 0.0f, 0.95f);
			if (lev > 30 || GPURandom::Rand(1.0f, state) > rr) return coef * mat->GetGlow();	
			coef = coef / rr;
		}
		return coef * mat->GetGlow() +
			RayTracing(Ray(hitpos + wi * 1e-4f, wi), root, m_Background, state, lev + 1, coef * att * fabs(glm::dot(wi, norm)) / poss);
	}
	else return coef * m_Background;
}


