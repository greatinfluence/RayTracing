#pragma once

#include "glm/vec3.hpp"

class Material {
public:
	Material(glm::vec3 glow): m_Glow(glow) {}
	virtual float scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
		glm::vec3& attenuation, glm::vec3& wi) = 0;
	glm::vec3 GetGlow() { return m_Glow; }
protected:
	glm::vec3 m_Glow;
};

class Diffuse : public Material {
public:
	Diffuse(): Material(glm::vec3(0.0f)), m_Albedo(glm::vec3(1.0f)) {}
	Diffuse(glm::vec3 albedo) : Material(glm::vec3(0.0f)), m_Albedo(albedo) {}
	float scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
		glm::vec3& attenuation, glm::vec3& wi) override;
private:
	glm::vec3 m_Albedo;
};
