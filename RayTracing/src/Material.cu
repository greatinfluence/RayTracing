#include "Material.h"

#include "Random.h"

#include <iostream>
//#include "Testtool.hpp"

__host__ __device__ float Diffuse::scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
    la::vec3& attenuation, la::vec3& wi, curandState* state) {
    // Using basic importance sampling
#ifdef __CUDA_ARCH__
    la::vec3 stdvec = GPURandom::RandinDisc(1.0f, *state);
#else
    la::vec3 stdvec = Random::RandinDisc(1.0f);
#endif
    la::vec3 perp = la::perp(norm + la::vec3(1.0f, 0, 0), norm);
    if(la::l2Norm(perp) < eps) [[unlikely]] {
        // norm = (1.0f, 0, 0)
        perp = la::vec3(0.0f, 1.0f, 0.0f);
    }
    else {
        perp = la::normalize(perp);
    }
    wi = norm * stdvec.y + perp * stdvec.x + la::cross(norm, perp) * stdvec.z;
    attenuation = m_Albedo / pi;
    return la::dot(wi, norm) / pi;
}

__host__ __device__ float Metal::scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
    la::vec3& attenuation, la::vec3& wi, curandState* state) {
    attenuation = m_Albedo;
#ifdef __CUDA_ARCH__
    wi = la::reflect(wo, norm) + GPURandom::RandinSphere(m_Fuzz, *state);
#else
    wi = la::reflect(wo, norm) + Random::RandinSphere(m_Fuzz);
#endif
    //if (m_Fuzz > 0.05f) {
     //   printf("(%f %f %f) - (%f %f %f) -> (%f %f %f)\n %f, %f, %f\n", wo.x, wo.y, wo.z, norm.x, norm.y, norm.z, wi.x, wi.y, wi.z, m_Fuzz, la::dot(wi, norm), la::dot(wo, norm));
   // }
    if (la::dot(wo, norm) < eps && la::dot(wi, norm) > -eps ||
        la::dot(wo, norm) > -eps && la::dot(wi, norm) < eps) {
        // Being absorbed
            attenuation = la::vec3(0.0f);
            wi = la::vec3(1.0f, 0, 0);
            return 1.0f;
    }
	wi = la::normalize(wi);
	float h = la::l2Norm(la::proj(wi, norm));
	if (h > m_Fuzz) {
		// Impossible to be absorbed
		return 1.0f;
	}
	else return (h + m_Fuzz) / (2 * m_Fuzz); // Compensate for the absorbed light
}

__host__ __device__ inline la::vec3 refract(la::vec3 wo, la::vec3 norm, float eta_ratio)
{
    float cosine = la::dot(wo, norm);
    la::vec3 rperp = (wo - norm * cosine) * (-eta_ratio);
    la::vec3 rpara = norm * ((float)((cosine > 0 ? -1 : 1) * (sqrt(1.0 - la::dot(rperp, rperp)))));
    return rperp + rpara;
}

__host__ __device__ inline float reflectance(float cosine, float ref_idx)
{
    float r0 = sq((1 - ref_idx) / (1 + ref_idx));
    return r0 + (1 - r0) * powf(1 - cosine, 5);
}

__host__ __device__ float Dieletric::scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
    la::vec3& attenuation, la::vec3& wi, curandState* state) {
    attenuation = la::vec3(1.0f);
    float cosine = la::dot(wo, norm);
    float sine = sqrtf(1.0 - cosine * cosine);
    float ref_idx = cosine > 0.0f ? (1.0f / m_Ir) : m_Ir;
    if (ref_idx * sine > 1.0f ||
#ifdef __CUDA_ARCH__
        GPURandom::Rand(1.0f, *state)
#else
        Random::Rand(1.0f)
#endif
        < reflectance(fabs(cosine), ref_idx)) {
        // Reflect
        wi = la::reflect(-wo, norm);
    }
    else {
        // Refract
         wi = refract(wo, norm, ref_idx);
    }
    return 1.0f;
}
