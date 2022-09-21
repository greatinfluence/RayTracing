#pragma once

#include "Random.h"

#include "la.h"

enum class MatType {
	None = 0,
	Diffuse = 1,
	Metal = 2,
	Dieletric = 3
};

class Material {
public:
	__host__ __device__ Material(la::vec3 glow): m_Glow(glow) {}

	// scatter(pos, wo, norm, attenuation, wi, state) will compute the scattered ray's direction
	//     and it's strength attenuation, save them into wi and attenuation.
	//     returns the possibility of the ray to be generated relative to uniform distribution
	__host__ __device__ virtual float scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
		la::vec3& attenuation, la::vec3& wi, curandState* state = nullptr) = 0;
	void SetGlow(la::vec3 glow) { m_Glow = glow; }
	__host__ __device__ la::vec3 GetGlow() const { return m_Glow; }
	virtual size_t GetSize() const = 0;
	virtual MatType GetType() const = 0;
protected:
	la::vec3 m_Glow;
};

class Diffuse : public Material {
public:
	Diffuse(): Material(la::vec3(0.0f)), m_Albedo(la::vec3(1.0f)) {}
	Diffuse(la::vec3 albedo) : Material(la::vec3(0.0f)), m_Albedo(albedo) {}
	__host__ __device__ Diffuse(la::vec3 gloom, la::vec3 albedo)
		: Material(gloom), m_Albedo(albedo) {}
	__host__ __device__ float scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
		la::vec3& attenuation, la::vec3& wi, curandState* state = nullptr) override;
	void SetAlbedo(la::vec3 albedo) { m_Albedo = albedo; }
	la::vec3 GetAlbedo() const { return m_Albedo; }
	MatType GetType() const override { return MatType::Diffuse; }
	size_t GetSize() const override { return sizeof(Diffuse); }
private:
	la::vec3 m_Albedo;
};

class Metal : public Material {
public:
	Metal(): Material(la::vec3(0.0f)), m_Albedo(la::vec3(1.0f)), m_Fuzz(0.0f) {}
	Metal(la::vec3 albedo): Material(la::vec3(0.0f)), m_Albedo(albedo), m_Fuzz(0.0f) {}
	Metal(la::vec3 albedo, float fuzz): Material(la::vec3(0.0f)), m_Albedo(albedo), m_Fuzz(fuzz) {}
	__host__ __device__ Metal(la::vec3 gloom, la::vec3 albedo, float fuzz)
		: Material(gloom), m_Albedo(albedo), m_Fuzz(fuzz) {}
	__host__ __device__ float scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
		la::vec3& attenuation, la::vec3& wi, curandState* state = nullptr) override;
	void SetAlbedo(la::vec3 albedo) { m_Albedo = albedo; }
	la::vec3 GetAlbedo() const { return m_Albedo; }
	float GetFuzz() const { return m_Fuzz; }
	MatType GetType() const override { return MatType::Metal; }
	size_t GetSize() const override { return sizeof(Metal); }
private:
	la::vec3 m_Albedo;
	float m_Fuzz;
};

class Dieletric : public Material {
public:
	Dieletric() : Material(la::vec3(0.0f)), m_Ir(1.0f) {}
	Dieletric(float ir) : Material(la::vec3(0.0f)), m_Ir(ir) {}
	__host__ __device__ Dieletric(la::vec3 gloom, float ir) : Material(gloom), m_Ir(ir) {}
	__host__ __device__ float scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
		la::vec3& attenuation, la::vec3& wi, curandState* state = nullptr) override;
	void SetIr(float ir) { m_Ir = ir; }
	float GetIr() const { return m_Ir; }
	MatType GetType() const override { return MatType::Dieletric; }
	size_t GetSize() const override { return sizeof(Dieletric); }
private:
	float m_Ir; // Index of refraction
};
