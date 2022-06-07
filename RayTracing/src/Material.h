#pragma once

#include "glm/vec3.hpp"

class Material {
public:
	Material(glm::vec3 glow): m_Glow(glow) {}
	virtual float scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
		glm::vec3& attenuation, glm::vec3& wi) = 0;
	void SetGlow(glm::vec3 glow) { m_Glow = glow; }
	glm::vec3 GetGlow() { return m_Glow; }
protected:
	glm::vec3 m_Glow;
};

class Diffuse : public Material {
public:
	Diffuse(): Material(glm::vec3(0.0f)), m_Albedo(glm::vec3(1.0f)) {}
	Diffuse(glm::vec3 albedo) : Material(glm::vec3(0.0f)), m_Albedo(albedo) {}
	Diffuse(glm::vec3 gloom, glm::vec3 albedo)
		: Material(gloom), m_Albedo(albedo) {}
	float scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
		glm::vec3& attenuation, glm::vec3& wi) override;
	void SetAlbedo(glm::vec3 albedo) { m_Albedo = albedo; }
	glm::vec3 GetAlbedo() { return m_Albedo; }
private:
	glm::vec3 m_Albedo;
};

class Metal : public Material {
public:
	Metal(): Material(glm::vec3(0.0f)), m_Albedo(glm::vec3(1.0f)), m_Fuzz(0.0f) {}
	Metal(glm::vec3 albedo): Material(glm::vec3(0.0f)), m_Albedo(albedo), m_Fuzz(0.0f) {}
	Metal(glm::vec3 albedo, float fuzz): Material(glm::vec3(0.0f)), m_Albedo(albedo), m_Fuzz(fuzz) {}
	Metal(glm::vec3 gloom, glm::vec3 albedo, float fuzz): Material(gloom), m_Albedo(albedo), m_Fuzz(fuzz) {}
	float scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
		glm::vec3& attenuation, glm::vec3& wi) override;
	void SetAlbedo(glm::vec3 albedo) { m_Albedo = albedo; }
	glm::vec3 GetAlbedo() { return m_Albedo; }
private:
	glm::vec3 m_Albedo;
	float m_Fuzz;
};
