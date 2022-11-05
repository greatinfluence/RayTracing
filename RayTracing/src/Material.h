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
	__host__ __device__ Material(MatType tpe, la::vec3 glow): m_Type(tpe), m_Glow(glow) {}

	// scatter(pos, wo, norm, attenuation, wi, state) will compute the scattered ray's direction
	//     and it's strength attenuation, save them into wi and attenuation.
	//     returns the possibility of the ray to be generated relative to uniform distribution
	//__host__ __device__ virtual float scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
	//	la::vec3& attenuation, la::vec3& wi, curandState* state = nullptr) = 0;
	void SetGlow(la::vec3 glow) { m_Glow = glow; }
	__host__ __device__ la::vec3 GetGlow() const { return m_Glow; }
	virtual size_t GetSize() const = 0;
	__host__ __device__ MatType GetType() const { return m_Type; }
protected:
	MatType m_Type;
	la::vec3 m_Glow;
};

class Diffuse : public Material {
public:
	Diffuse(): Material(MatType::Diffuse, la::vec3(0.0f)), m_Albedo(la::vec3(1.0f)) {}
	Diffuse(la::vec3 albedo) : Material(MatType::Diffuse, la::vec3(0.0f)), m_Albedo(albedo) {}
	__host__ __device__ Diffuse(la::vec3 gloom, la::vec3 albedo)
		: Material(MatType::Diffuse, gloom), m_Albedo(albedo) {}
	__host__ __device__ float scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
		la::vec3& attenuation, la::vec3& wi, curandState* state = nullptr);
	void SetAlbedo(la::vec3 albedo) { m_Albedo = albedo; }
	la::vec3 GetAlbedo() const { return m_Albedo; }
	size_t GetSize() const override { return sizeof(Diffuse); }
private:
	la::vec3 m_Albedo;
};

class Metal : public Material {
public:
	Metal(): Material(MatType::Metal, la::vec3(0.0f)), m_Albedo(la::vec3(1.0f)), m_Fuzz(0.0f) {}
	Metal(la::vec3 albedo): Material(MatType::Metal, la::vec3(0.0f)), m_Albedo(albedo), m_Fuzz(0.0f) {}
	Metal(la::vec3 albedo, float fuzz): Material(MatType::Metal, la::vec3(0.0f)), m_Albedo(albedo), m_Fuzz(fuzz) {}
	__host__ __device__ Metal(la::vec3 gloom, la::vec3 albedo, float fuzz)
		: Material(MatType::Metal, gloom), m_Albedo(albedo), m_Fuzz(fuzz) {}
	__host__ __device__ float scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
		la::vec3& attenuation, la::vec3& wi, curandState* state = nullptr);
	void SetAlbedo(la::vec3 albedo) { m_Albedo = albedo; }
	la::vec3 GetAlbedo() const { return m_Albedo; }
	float GetFuzz() const { return m_Fuzz; }
	size_t GetSize() const override { return sizeof(Metal); }
private:
	la::vec3 m_Albedo;
	float m_Fuzz;
};

class Dieletric : public Material {
public:
	Dieletric() : Material(MatType::Dieletric, la::vec3(0.0f)), m_Ir(1.0f) {}
	Dieletric(float ir) : Material(MatType::Dieletric, la::vec3(0.0f)), m_Ir(ir) {}
	__host__ __device__ Dieletric(la::vec3 gloom, float ir) : Material(MatType::Dieletric, gloom), m_Ir(ir) {}
	__host__ __device__ float scatter(la::vec3 pos, la::vec3 wo, la::vec3 norm,
		la::vec3& attenuation, la::vec3& wi, curandState* state = nullptr);
	void SetIr(float ir) { m_Ir = ir; }
	float GetIr() const { return m_Ir; }
	size_t GetSize() const override { return sizeof(Dieletric); }
private:
	float m_Ir; // Index of refraction
};
