#include "Material.h"

#include "Random.h"

#include "glm/glm.hpp"
#include "glm/gtx/perpendicular.hpp"

#include <iostream>

float Diffuse::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm, glm::vec3& attenuation,
    glm::vec3& wi)
{
    wi = Random::RandinDisc(1.0f);
    glm::vec3 ppnm = glm::perp(norm + glm::vec3(1.0f, 0, 0), norm);
    if (glm::l2Norm(ppnm) < 1e-6f) {
        // norm = (1, 0, 0)
        ppnm = glm::vec3(0, 1.0f, 0);
    }
    else ppnm = glm::normalize(ppnm);
    auto fi = wi;
    wi = wi.y * norm + wi.x * ppnm + wi.z * glm::cross(norm, ppnm);
    if (fabs(wi.x) > 1.5f || fabs(wi.y) > 1.5f || fabs(wi.z) > 1.5f) {
        std::cout << "NaN" << std::endl;
    }
    attenuation = m_Albedo / pi;
    return glm::dot(norm, wi) / pi;
}

float Metal::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm, glm::vec3& attenuation, glm::vec3& wi)
{
    attenuation = m_Albedo;
    wi = glm::reflect(-wo, norm) + Random::RandinSphere(m_Fuzz);
    if (glm::dot(wi, norm) < 1e-6) {
        // wi = 0 or too small angle
        attenuation = glm::vec3(0.0f);
    }
    wi = glm::normalize(wi);
    return 1.0f;
}
