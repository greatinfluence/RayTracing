#pragma once

#include "lasetting.h"
#include <cmath>

namespace la {
	struct vec3 {
		float x, y, z;

		FUNCPRE vec3() : x{ 0 }, y{ 0 }, z{ 0 } {}
		FUNCPRE vec3(float f) : x{ f }, y{ f }, z{ f } {}
		FUNCPRE vec3(float x, float y, float z) : x{ x }, y{ y }, z{ z } {}

		FUNCPRE float operator[](int ind) const;
		FUNCPRE vec3 operator+(vec3 const& rhs) const;
		FUNCPRE vec3 operator-(vec3 const& rhs) const;
		FUNCPRE vec3 operator*(float fac) const;
		FUNCPRE vec3 operator*(vec3 const& rhs) const;
		FUNCPRE vec3& operator+=(vec3 const& rhs);
		FUNCPRE vec3& operator*=(float fac);
		FUNCPRE vec3& operator/=(float fac);
		FUNCPRE vec3 operator/(float fac) const;
		FUNCPRE vec3 operator-() const;
		FUNCPRE friend vec3 operator*(float fac, vec3 const& v);
	};

	// dot(u, v) returns <u, v>.
	FUNCPRE float dot(vec3 const& u, vec3 const& v);

	// l2Norm(u) returns the l2 norm of u.
	FUNCPRE float l2Norm(vec3 const& u);

	// proj(u, v) returns the projection of u over v.
	FUNCPRE vec3 proj(vec3 const& u, vec3 const& v);

	// perp(u, v) returns the perpendicular component of u with respect to v.
	FUNCPRE vec3 perp(vec3 const& u, vec3 const& v);

	// normalize(u) returns the normalized direction vector of u.
	FUNCPRE vec3 normalize(vec3 const& u);

	// cross(u, v) returns the cross product of u and v.
	FUNCPRE vec3 cross(vec3 const& u, vec3 const& v);

	// reflect(u, v) returns the reflected vector of u with respect to v.
	FUNCPRE vec3 reflect(vec3 const& u, vec3 const& v);

	// clamp(x, mn, mx) returns min(max(x, mn), mx).
	FUNCPRE float clamp(float x, float mn, float mx);

	// clamp(x, mn, mx) returns min(max(x, mn), mx).
	FUNCPRE vec3 clamp(vec3 x, vec3 mn, vec3 mx);

	// sqrt(v) returns (sqrt(v.x), sqrt(v.y), sqrt(v.z)).
	FUNCPRE vec3 sqrt(vec3 const& v);
}

namespace la {
	FUNCPRE float vec3::operator[](int ind) const
	{
		return *(& x + ind);
	}
	FUNCPRE vec3 vec3::operator+(vec3 const& rhs) const
	{
		return vec3(x + rhs.x, y + rhs.y, z + rhs.z);
	}
	FUNCPRE vec3 vec3::operator-(vec3 const& rhs) const
	{
		return vec3(x - rhs.x, y - rhs.y, z - rhs.z);
	}
	FUNCPRE vec3 vec3::operator*(float fac) const
	{
		return vec3(fac * x, fac * y, fac * z);
	}
	FUNCPRE vec3 vec3::operator*(vec3 const& rhs) const
	{
		return vec3(x * rhs.x, y * rhs.y, z * rhs.z);
	}
	FUNCPRE vec3& vec3::operator+=(vec3 const& rhs)
	{
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
		return *this;
	}
	FUNCPRE vec3& vec3::operator*=(float fac)
	{
		x *= fac;
		y *= fac;
		z *= fac;
		return *this;
	}
	FUNCPRE vec3& vec3::operator/=(float fac)
	{
		x /= fac;
		y /= fac;
		z /= fac;
		return *this;
	}
	FUNCPRE vec3 vec3::operator/(float fac) const
	{
		return vec3(x / fac, y / fac, z / fac);
	}
	FUNCPRE vec3 vec3::operator-() const
	{
		return vec3(-x, -y, -z);
	}
	FUNCPRE vec3 operator*(float fac, vec3 const& v)
	{
		return vec3(fac * v.x, fac * v.y, fac * v.z);
	}
	FUNCPRE float dot(vec3 const& u, vec3 const& v)
	{
		return u.x * v.x + u.y * v.y + u.z * v.z;
	}
	FUNCPRE float l2Norm(vec3 const& u)
	{
		return sqrtf(dot(u, u));
	}
	FUNCPRE vec3 proj(vec3 const& u, vec3 const& v)
	{
		return v * (dot(u, v) / dot(v, v));
	}
	FUNCPRE vec3 perp(vec3 const& u, vec3 const& v)
	{
		return u - proj(u, v);
	}
	FUNCPRE vec3 normalize(vec3 const& u)
	{
		return u / l2Norm(u);
	}
	FUNCPRE vec3 cross(vec3 const& u, vec3 const& v)
	{
		return vec3(u.y * v.z - v.y * u.z, u.z * v.x - v.z * u.x, u.x * v.y - v.x * u.y);
	}
	FUNCPRE vec3 reflect(vec3 const& u, vec3 const& v)
	{
		return u - perp(u, v) * 2;
	}
	FUNCPRE float clamp(float x, float mn, float mx)
	{
		return fmin(fmax(x, mn), mx);
	}
	FUNCPRE vec3 clamp(vec3 x, vec3 mn, vec3 mx)
	{
		return vec3(clamp(x.x, mn.x, mx.x), clamp(x.y, mn.y, mx.y), clamp(x.z, mn.z, mx.z));
	}
	FUNCPRE vec3 sqrt(vec3 const& v)
	{
		return vec3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
	}
}

