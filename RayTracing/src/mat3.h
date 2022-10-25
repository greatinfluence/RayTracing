#pragma once

#include "lasetting.h"
#include "vec3.h"

namespace la {
	struct mat3 {
		float comp[3][3];

		// returns I3
		FUNCPRE mat3() : comp{ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} } {}
		
		// returns f * I3
		FUNCPRE mat3(float f): comp{{f, 0, 0}, {0, f, 0}, {0, 0, f}} {}

		FUNCPRE mat3(float mat[][3]) {
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					comp[i][j] = mat[i][j];
		}

		// returns x00, 0y0, 00z
		FUNCPRE mat3(float x, float y, float z) : comp{ {x, 0, 0}, {0, y, 0}, {0, 0, z} } {}

		struct Col {
			float* ptr;
			float operator[](int ind) const { return *(ptr + ind); }
		} c[3]{ {comp[0]}, {comp[1]}, {comp[2]} };
		FUNCPRE Col const& operator[](int ind) const;
		FUNCPRE mat3 operator+(mat3 const& rhs) const;
		FUNCPRE mat3 operator-(mat3 const& rhs) const;
		FUNCPRE mat3 operator*(mat3 const& rhs) const;
		FUNCPRE vec3 operator*(vec3 const& rhs) const;
		FUNCPRE mat3 operator*(float f) const;
		FUNCPRE mat3& operator+=(mat3 const& rhs);
		FUNCPRE mat3& operator-=(mat3 const& rhs);
		FUNCPRE mat3& operator*=(mat3 const& rhs);
		FUNCPRE mat3& operator*=(float f);
		FUNCPRE mat3 operator-() const;
		FUNCPRE mat3 T() const;
		FUNCPRE friend mat3 operator*(float fac, mat3 const& m);
		FUNCPRE friend vec3 operator*(vec3 const& v, mat3 const& m);
	};
}

namespace la {
	FUNCPRE mat3::Col const& mat3::operator[](int ind) const {
		return c[ind];
	}
	FUNCPRE mat3  mat3::operator+(mat3 const& rhs) const {
		float ans[3][3];
		for (int i = 0; i < 3; ++i) {
			float const* l = comp[i];
			float const* r = rhs.comp[i];
			float* a = ans[i];
			a[0] = l[0] + r[0];
			a[1] = l[1] + r[1];
			a[2] = l[2] + r[2];
		}
		return mat3(ans);
	}
	FUNCPRE mat3  mat3::operator-(mat3 const& rhs) const {
		float ans[3][3] = { 0 };
		for (int i = 0; i < 3; ++i) {
			float const* l = comp[i];
			float const* r = rhs.comp[i];
			float* a = ans[i];
			a[0] = l[0] - r[0];
			a[1] = l[1] - r[1];
			a[2] = l[2] - r[2];
		}
		return mat3(ans);
	}
	FUNCPRE mat3  mat3::operator*(mat3 const& rhs) const {
		float ans[3][3] = { 0 };
		for (int k = 0; k < 3; ++k) {
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					ans[i][j] += comp[i][k] * rhs.comp[k][j];
				}
			}
		}
		return mat3(ans);
	}
	FUNCPRE vec3  mat3::operator*(vec3 const& rhs) const {
		float ans[3] = { 0 };
		for(int i = 0; i < 3; ++ i)
			for (int j = 0; j < 3; ++j) {
				ans[i] += comp[i][j] * rhs[j];
			}
		return vec3(ans[0], ans[1], ans[2]);

	}
	FUNCPRE mat3  mat3::operator*(float f) const {
		float ans[3][3];
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) ans[i][j] = f * comp[i][j];
		return mat3(ans);
	}
	FUNCPRE mat3& mat3::operator+=(mat3 const& rhs) {
		for (int i = 0; i < 3; ++i) {
			float* l = comp[i];
			float const* r = rhs.comp[i];
			l[0] += r[0];
			l[1] += r[1];
			l[2] += r[2];
		}
	}
	FUNCPRE mat3& mat3::operator-=(mat3 const& rhs) {
		for (int i = 0; i < 3; ++i) {
			float* l = comp[i];
			float const* r = rhs.comp[i];
			l[0] -= r[0];
			l[1] -= r[1];
			l[2] -= r[2];
		}
	}
	FUNCPRE mat3& mat3::operator*=(mat3 const& rhs) {
		float ans[3][3] = { 0 };
		for (int k = 0; k < 3; ++k) {
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					ans[i][j] += comp[i][k] * rhs.comp[k][j];
				}
			}
		}
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) comp[i][j] = ans[i][j];

	}
	FUNCPRE mat3& mat3::operator*=(float f) {
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) comp[i][j] *= f;
	}
	FUNCPRE mat3 mat3::operator-() const {
		float ans[3][3];
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) ans[i][j] = -comp[i][j];
		return mat3(ans);
	}
	FUNCPRE mat3 mat3::T() const {
		float ans[3][3];
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) ans[i][j] = comp[j][i];
		return mat3(ans);
	}
	FUNCPRE mat3 operator*(float fac, mat3 const& m) {
		return m * fac;
	}
	FUNCPRE vec3 operator*(vec3 const& v, mat3 const& m) {
		float ans[3] = { 0 };
		for(int i = 0; i < 3; ++ i)
			for (int j = 0; j < 3; ++j) {
				ans[i] += v[j] * m.comp[j][i];
			}
		return vec3(ans[0], ans[1], ans[2]);
	}
}
