#include "Raytracer.h"

#include "Geometryrepository.h"

__device__ la::vec3 Raytracing::RayTracing(Ray const& ray, Cuboid* cub, la::vec3 const& m_Background,
	curandState& state, int lev, la::vec3 coef) {
	float dist = floatmax;
	Geometry const* hitted = nullptr;
	cub->TestHitdevice(ray, dist, hitted);
	if (hitted != nullptr) {
		la::vec3 hitpos = ray.m_Pos + ray.m_Dir * dist;
		la::vec3 att, wi, norm = hitted->GetNorm(hitpos);
		Material* mat = hitted->GetMaterial();
		float poss = mat->scatter(hitpos, -ray.m_Dir,
			norm, att, wi, &state);
	//	if (fabs(dist) > 1e3 || fabs(ray.m_Dir.x) > 1.5f) {
	//		// Impossible!
	//		printf("What?\n");
	//	}
		if(lev > 4) {
			float rr = la::clamp(fmax(coef.x, fmax(coef.y, coef.z)), 0.0f, 0.95f);
			if (lev > 30 || GPURandom::Rand(1.0f, state) > rr) return coef * mat->GetGlow();
			coef /= rr;
		}
	//	if (ret.x < 0.5 || ret.y < 0.5 || ret.z < 0.5) {
		//	printf("(%f, %f, %f), %d\n", ret.x, ret.y, ret.z, lev);
		//}
		return coef * mat->GetGlow() +
			RayTracing(Ray(hitpos + wi * 1e-4f, wi), cub, m_Background, state, lev + 1, coef * att * fabs(la::dot(wi, norm)) / poss);
	}
	else return coef * m_Background;
}


