#pragma once

#include "glm/vec3.hpp"
#include "Random.h"

enum class MatType {
	None = 0,
	Diffuse = 1,
	Metal = 2,
	Dieletric = 3
};

class Material {
public:
	Material(glm::vec3 glow): m_Glow(glow) {}

	// scatter(pos, wo, norm, attenuation, wi, state) will compute the scattered ray's direction
	//     and it's strength attenuation, save them into wi and attenuation.
	//     returns the possibility of the ray to be generated relative to uniform distribution
	__host__ __device__ virtual float scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
		glm::vec3& attenuation, glm::vec3& wi, curandState* state = nullptr) = 0;
	void SetGlow(glm::vec3 glow) { m_Glow = glow; }
	__host__ __device__ glm::vec3 GetGlow() const { return m_Glow; }
	virtual size_t GetSize() const = 0;
	virtual MatType GetType() const = 0;
protected:
	glm::vec3 m_Glow;
};

class Diffuse : public Material {
public:
	Diffuse(): Material(glm::vec3(0.0f)), m_Albedo(glm::vec3(1.0f)) {}
	Diffuse(glm::vec3 albedo) : Material(glm::vec3(0.0f)), m_Albedo(albedo) {}
	Diffuse(glm::vec3 gloom, glm::vec3 albedo)
		: Material(gloom), m_Albedo(albedo) {}
	__host__ __device__ float scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
		glm::vec3& attenuation, glm::vec3& wi, curandState* state = nullptr) override;
	void SetAlbedo(glm::vec3 albedo) { m_Albedo = albedo; }
	glm::vec3 GetAlbedo() const { return m_Albedo; }
	MatType GetType() const override { return MatType::Diffuse; }
	size_t GetSize() const override { return sizeof(Diffuse); }
private:
	glm::vec3 m_Albedo;
};

class Metal : public Material {
public:
	Metal(): Material(glm::vec3(0.0f)), m_Albedo(glm::vec3(1.0f)), m_Fuzz(0.0f) {}
	Metal(glm::vec3 albedo): Material(glm::vec3(0.0f)), m_Albedo(albedo), m_Fuzz(0.0f) {}
	Metal(glm::vec3 albedo, float fuzz): Material(glm::vec3(0.0f)), m_Albedo(albedo), m_Fuzz(fuzz) {}
	Metal(glm::vec3 gloom, glm::vec3 albedo, float fuzz): Material(gloom), m_Albedo(albedo), m_Fuzz(fuzz) {}
	__host__ __device__ float scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
		glm::vec3& attenuation, glm::vec3& wi, curandState* state = nullptr) override;
	void SetAlbedo(glm::vec3 albedo) { m_Albedo = albedo; }
	glm::vec3 GetAlbedo() const { return m_Albedo; }
	float GetFuzz() const { return m_Fuzz; }
	MatType GetType() const override { return MatType::Metal; }
	size_t GetSize() const override { return sizeof(Metal); }
private:
	glm::vec3 m_Albedo;
	float m_Fuzz;
};

class Dieletric : public Material {
public:
	Dieletric() : Material(glm::vec3(0.0f)), m_Ir(1.0f) {}
	Dieletric(float ir) : Material(glm::vec3(0.0f)), m_Ir(ir) {}
	Dieletric(glm::vec3 gloom, float ir) : Material(gloom), m_Ir(ir) {}
	__host__ __device__ float scatter(glm::vec3 pos, glm::vec3 wo, glm::vec3 norm,
		glm::vec3& attenuation, glm::vec3& wi, curandState* state = nullptr) override;
	void SetIr(float ir) { m_Ir = ir; }
	float GetIr() const { return m_Ir; }
	MatType GetType() const override { return MatType::Dieletric; }
	size_t GetSize() const override { return sizeof(Dieletric); }
private:
	float m_Ir; // Index of refraction

	// refract(wo, norm, eta_ratio) computes the refracted light's direction
	glm::vec3 refract(glm::vec3 wo, glm::vec3 norm, float eta_ratio);

	// reflectance(cosine, ref_idx) computes the reflectance rate using Schlick's approximation
	float reflectance(float cosine, float ref_idx);
};
