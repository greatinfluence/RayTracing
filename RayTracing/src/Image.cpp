#include "Image.h"

#include <iostream>

#include "stb_Image/stb_Image.h"
#include "stb_Image/stb_Image_write.h"

Image3::Image3(int width, int height, std::string filepath, int channels, int deschannels)
	: m_Width(width), m_Height(height), m_Channels(deschannels)
{
	if (filepath != "") {
		m_Data = stbi_load(filepath.c_str(), &m_Width, &m_Height,
			&channels, deschannels);
		if (m_Data == nullptr) {
			std::cout << "Unable to load the Image " + filepath << std::endl;
		}
	}
	else
	{
		m_Data = new unsigned char[width * height * channels];
	}
}

Image3::~Image3()
{
	delete[] m_Data;
}

void Image3::Setcol(int x, int y, glm::vec3 col, bool regularize)
{
	if (regularize) {
		col *= 255;
	}
	y = m_Height - 1 - y; // Flip the paint
	unsigned char* pos = m_Data + (y * m_Width + x) * m_Channels;
	*pos = (unsigned char)round(col.r);
	*(pos + 1) = (unsigned char)round(col.g);
	*(pos + 2) = (unsigned char)round(col.b);
}

glm::vec3 Image3::Readcol(int x, int y, bool regularize)
{
	y = m_Height - 1 - y; // Flip the paint
	unsigned char* pos = m_Data + (y * m_Width + x) * m_Channels;
	auto col = glm::vec3(*pos, *(pos + 1), *(pos + 2));
	if (regularize) {
		col /= 255;
	}
	return col;
}

void Image3::Write(std::string filepath)
{
	if (str_ends_in(filepath, ".png") || str_ends_in(filepath, ".PNG")) {
		stbi_write_png(filepath.c_str(), m_Width, m_Height, m_Channels, m_Data,
			m_Width * m_Channels);
	}
	else if (str_ends_in(filepath, ".jpg") || str_ends_in(filepath, ".JPG") ||
		str_ends_in(filepath, ".jpeg") || str_ends_in(filepath, ".JPEG")) {
		stbi_write_jpg(filepath.c_str(), m_Width, m_Height, m_Channels, m_Data, 100);
	}
}

bool Image3::str_ends_in(std::string source, std::string suffix)
{
	size_t slen = source.length(), sflen = suffix.length();
	if (slen < sflen) return false;
	return source.compare(slen - sflen, sflen, suffix);
}
