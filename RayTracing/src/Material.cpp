#include "Material.h"

#include "Random.h"

#include "glm/glm.hpp"
#include "glm/gtx/perpendicular.hpp"

float Diffuse::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm, glm::vec3& attenuation,
    glm::vec3& wi)
{
    wi = Random::GenVec(1.0f);
    glm::vec3 ppnm = glm::normalize(glm::perp(norm + glm::vec3(0, 1, 0), norm));
    if (glm::dot(ppnm, norm) < 0.95f) {
        ppnm = glm::normalize(glm::perp(norm + glm::vec3(1, 0, 0), norm));
    }
    wi = wi.y * norm + wi.x * ppnm + wi.z * glm::cross(norm, ppnm);
    attenuation = m_Albedo / pi;
    return glm::dot(norm, wi) / pi;
}
