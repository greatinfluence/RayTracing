#pragma once

#include <string>

#include "glm/vec3.hpp"

class Image3 {
public:
	Image3(int width, int height, std::string filepath = "", int channels = 3, int deschannels = 3);
	~Image3();
	void Setcol(int x, int y, glm::vec3 col, bool regularize = false);
	glm::vec3 Readcol(int x, int y, bool regularize = false);
	void Write(std::string filepath);
private:
	int m_Width, m_Height, m_Channels;
	unsigned char* m_Data;

	bool str_ends_in(std::string source, std::string suffix);
};