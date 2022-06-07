#include "Material.h"

#include "Random.h"

#include "glm/glm.hpp"
#include "glm/gtx/perpendicular.hpp"

float Diffuse::scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm, glm::vec3& attenuation,
    glm::vec3& wi)
{
    wi = Random::GenVec(1.0f);
    glm::vec3 ppnm = glm::perp(norm + glm::vec3(1, 0, 0), norm);
    if (glm::l2Norm(ppnm) < 1e-6f) {
        // norm = (1, 0, 0)
        ppnm = glm::vec3(0, 1.0f, 0);
    }
    else ppnm = glm::normalize(ppnm);
    wi = wi.y * norm + wi.x * ppnm + wi.z * glm::cross(norm, ppnm);
    attenuation = m_Albedo / pi;
    return glm::dot(norm, wi) / pi;
}
