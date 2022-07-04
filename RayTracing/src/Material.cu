#include "Material.h"

#include "Random.h"

#include "glm/glm.hpp"
#include "glm/gtx/perpendicular.hpp"

#include <iostream>
//#include "Testtool.hpp"

__host__ __device__ float Diffuse::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
    glm::vec3& attenuation, glm::vec3& wi, curandState* state) {
    // Using basic importance sampling
#ifdef __CUDA_ARCH__
    glm::vec3 stdvec = GPURandom::RandinDisc(1.0f, *state);
#else
    glm::vec3 stdvec = Random::RandinDisc(1.0f);
#endif
    glm::vec3 perp = glm::perp(norm, norm + glm::vec3(1.0f, 0, 0));
    if(glm::l2Norm(perp) < eps) {
        // norm = (1.0f, 0, 0)
        perp = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else {
        perp = glm::normalize(perp);
    }
    wi = norm * stdvec.y + perp * stdvec.x + glm::cross(norm, perp) * stdvec.z;
    attenuation = m_Albedo / pi;
    return glm::dot(wi, norm) / pi;
}

__host__ __device__ float Metal::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
    glm::vec3& attenuation, glm::vec3& wi, curandState* state) {
    attenuation = m_Albedo;
#ifdef __CUDA_ARCH__
    wi = glm::reflect(-wo, norm) + GPURandom::RandinSphere(m_Fuzz, *state);
#else
    wi = glm::reflect(-wo, norm) + Random::RandinSphere(m_Fuzz);
#endif
    if (glm::dot(wo, norm) < eps && glm::dot(wi, norm) > -eps ||
        glm::dot(wo, norm) > -eps && glm::dot(wi, norm) < eps) {
        // Being absorbed
            attenuation = glm::vec3(0.0f);
            wi = glm::vec3(1.0f, 0, 0);
            return 1.0f;
    }
	wi = glm::normalize(wi);
	float h = glm::l2Norm(glm::proj(wi, norm));
	if (h > m_Fuzz) {
		// Impossible to be absorbed
		return 1.0f;
	}
	else return (h + m_Fuzz) / (2 * m_Fuzz); // Compensate for the absorbed light
}

__host__ __device__ float Dieletric::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
    glm::vec3& attenuation, glm::vec3& wi, curandState* state) {
    attenuation = glm::vec3(1.0f);
    float cosine = glm::dot(wo, norm);
    float sine = (float)sqrt(1.0 - cosine * cosine);
    float ref_idx = cosine > 0.0f ? (1.0f / m_Ir) : m_Ir;
    if (ref_idx * sine > 1.0f ||
#ifdef __CUDA_ARCH__
        GPURandom::Rand(1.0f, *state)
#else
        Random::Rand(1.0f)
#endif
        < reflectance(abs(cosine), ref_idx)) {
        // Reflect
        wi = glm::reflect(-wo, norm);
    }
    else {
        // Refract
         wi = refract(wo, norm, ref_idx);
    }
    return 1.0f;
}

glm::vec3 Dieletric::refract(glm::vec3 wo, glm::vec3 norm, float eta_ratio)
{
    float cosine = glm::dot(wo, norm);
    glm::vec3 rperp = -eta_ratio * (wo - cosine * norm);
    glm::vec3 rpara = ((float)((cosine > 0 ? -1 : 1) * (sqrt(1.0 - glm::dot(rperp, rperp))))) * norm;
    return rperp + rpara;
}

float Dieletric::reflectance(float cosine, float ref_idx)
{
    float r0 = sq((1 - ref_idx) / (1 + ref_idx));
    return (float)(r0 + (1 - r0) * pow(1 - cosine, 5));
}
