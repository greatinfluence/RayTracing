#include "Material.h"

#include "Random.h"

#include "glm/glm.hpp"
#include "glm/gtx/perpendicular.hpp"

#include <iostream>
//#include "Testtool.hpp"

float Diffuse::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm, glm::vec3& attenuation,
    glm::vec3& wi)
{
    // Using basic importance sampling
    auto stdvec = Random::RandinDisc(1.0f);
    auto perp = glm::perp(norm, norm + glm::vec3(1.0f, 0, 0));
    if(glm::l2Norm(perp) < 1e-6) {
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

float Metal::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm, glm::vec3& attenuation, glm::vec3& wi)
{
    attenuation = m_Albedo;
    if (glm::dot(wo, norm) < 0) {
        // Inside the ball
        wi = glm::reflect(-wo, norm) + Random::RandinSphere(m_Fuzz);
        if (glm::dot(wi, norm) > -1e-6) {
			// wi = 0 or too small angle
            attenuation = glm::vec3(0.0f);
            wi = glm::vec3(1.0f, 0, 0);
            return 1.0f;
        }
        wi = glm::normalize(wi);
        auto h = glm::l2Norm(glm::proj(wi, norm));
        if (h > m_Fuzz - 1e-6) {
            // No possible to be canceled
            return 1.0f;
        }
        else return  (h + m_Fuzz) / (2 * m_Fuzz);
    }
    else {
        // Outside the ball
		wi = glm::reflect(-wo, norm) + Random::RandinSphere(m_Fuzz);
		if (glm::dot(wi, norm) < 1e-6) {
			// wi = 0 or too small angle
			attenuation = glm::vec3(0.0f);
			wi = glm::vec3(1.0f, 0, 0);
			return 1.0f;
		}
		wi = glm::normalize(wi);
        auto h = glm::l2Norm(glm::proj(wi, norm));
        if (h > m_Fuzz - 1e-6) {
            // No possible to be canceled
            return 1.0f;
        }
        else return  (h + m_Fuzz) / (2 * m_Fuzz);
    }
}

float Dieletric::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm, glm::vec3& attenuation, glm::vec3& wi)
{
    attenuation = glm::vec3(1.0f);
    float cosine = glm::dot(wo, norm);
    float sine = sqrt(1.0 - cosine * cosine);
    float ref_idx = cosine > 0.0 ? (1.0 / m_Ir) : m_Ir;
    float poss = ref_idx * sine > 1.0f ? 1 : reflectance(fabs(cosine), ref_idx);
    if (ref_idx * sine > 1.0f || Random::Rand(1.0f) < poss) {
        // Reflect
        wi = glm::reflect(-wo, norm);
        //assert(glm::dot(wi, norm) > -1e-6);
        //assert(fabs(glm::l2Norm(wi) - 1) < 1e-6);
       //return poss;
    }
    else {
        // Refract
         wi = refract(wo, norm, ref_idx);
       // return 1 - poss;
    }
    return 1.0f;
}

glm::vec3 Dieletric::refract(glm::vec3 wo, glm::vec3 norm, float eta_ratio)
{
    auto cosine = glm::dot(wo, norm);
    glm::vec3 rperp = -eta_ratio * (wo - cosine * norm);
    glm::vec3 rpara = ((float)((cosine > 0 ? -1 : 1) * (sqrt(1.0 - glm::dot(rperp, rperp))))) * norm;
    return rperp + rpara;
}

float Dieletric::reflectance(float cosine, float ref_idx)
{
    //return 0.0f;
    // Schlick's approximation for reflectance
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow(1 - cosine, 5);
}
