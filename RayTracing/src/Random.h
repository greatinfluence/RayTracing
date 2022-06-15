#pragma once

#include <Random>

#include <random>

#include "glm/gtx/norm.hpp"
#include "glm/vec3.hpp"
#include "Settings.h"

class Random {
public:
	// Rand(l, r) produces uniformly scattered random numbers in [l, r]
	inline static float Rand(float l, float r) { return GetInstance().prRand(l, r); }
	
	// Rand(r) produces uniformly scattered random numbers in [0, r]
	inline static float Rand(float r) { return GetInstance().prRand(r); }

	// RandinDisc(r) produces 3D vectors with x, y axis randomly scattered in the disc with radius r
	inline static glm::vec3 RandinDisc(float r) { return GetInstance().prRandinDisc(r); }

	// RandinSphere(r) produces 3D vectors randomly scattered in the sphere with radius r
	inline static glm::vec3 RandinSphere(float r) { return GetInstance().prRandinSphere(r); }

	// RandinHemisphere(norm, r) produces 3D vectors randomly scattered in the hemisphere with radius r, normal direction norm
	inline static glm::vec3 RandinHemisphere(glm::vec3 norm, float r) {
		auto vec = RandinSphere(r);
		if (glm::dot(vec, norm) < 0) return -vec;
		return vec;
	}
private:
	Random() = default;
	Random(Random const&) = delete;
	Random(Random&&) = delete;
	Random& operator=(Random const&) = delete;
	Random& operator=(Random&&) = delete;
	inline float prRand(float l, float r) {
		std::uniform_real_distribution<float> dist(l, r);
		return (float)dist(m_Engine);
	}
	inline float prRand(float r) {
		std::uniform_real_distribution<float> dist(0, r);
		return (float)dist(m_Engine);
	}
	inline glm::vec3 prRandinDisc(float r) {
		float theta = prRand(2 * pi);
		float rad = prRand(r);
		rad = sqrt(rad);
		return glm::vec3(rad * cos(theta), sqrt(sq(r) - sq(rad)), rad * sin(theta));
	}

	inline glm::vec3 prRandinSphere(float r) {
		float theta = prRand(2 * pi), phi = prRand(pi);
		float rad = prRand(r);
		rad = sqrt(rad);
		return glm::vec3(rad * cos(phi) * cos(theta), rad * cos(phi) * sin(theta), rad * sin(phi));
	}
	inline static Random& GetInstance() {
		static Random* Instance = new Random;
		return *Instance;
	}
	std::mt19937 m_Engine;
};
