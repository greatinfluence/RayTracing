#pragma once

#include <Random>

#include <random>

#include "glm/gtx/norm.hpp"
#include "glm/vec3.hpp"

float constexpr pi = 3.14159265358979323846f;

class Random {
public:
	inline static float Rand(float l, float r) { return GetInstance().prRand(l, r); }
	inline static float Rand(float r) { return GetInstance().prRand(r); }
	inline static glm::vec3 GenVec(float r) { return GetInstance().prGenVec(r); }
private:
	Random() = default;
	Random(Random const&) = delete;
	Random(Random&&) = delete;
	Random& operator=(Random const&) = delete;
	Random& operator=(Random&&) = delete;
	inline float prRand(float l, float r) {
		std::uniform_real_distribution<> dist(l, r);
		return (float)dist(m_Engine);
	}
	inline float prRand(float r) {
		std::uniform_real_distribution<> dist(0, r);
		return (float)dist(m_Engine);
	}
	inline glm::vec3 prGenVec(float r) {
		while(true) {
		    glm::vec3 vec(Rand(r), Rand(r), 0.0);
		    if (glm::l2Norm(vec) > r) continue;
		    return glm::vec3(vec.x, vec.y, sqrt(1.0f - vec.x * vec.x - vec.y * vec.y));
		}
	}
	inline static Random& GetInstance() {
		static Random* Instance = new Random;
		return *Instance;
	}
	std::mt19937 m_Engine;
};
