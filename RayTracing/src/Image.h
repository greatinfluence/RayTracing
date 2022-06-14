#pragma once

#include <string>

#include "glm/vec3.hpp"

class Image3 {
public:
	Image3(): m_Width(32), m_Height(16), m_Channels(0), m_Filepath(""),
		m_Data(nullptr) {}
	Image3(int width, int height, std::string filepath = "", int deschannels = 3);
	Image3(Image3 const& img) noexcept;
	Image3& operator=(Image3&& img) noexcept;
	Image3& operator=(Image3 const& img) noexcept;
	~Image3();
	void Setcol(int x, int y, glm::vec3 col, bool regularize = false);
	glm::vec3 Readcol(int x, int y, bool regularize = false);
	void Write(std::string filepath);
	int GetWidth() const { return m_Width; }
	int GetHeight() const { return m_Height; }
	int GetChannels() const { return m_Channels; }
	std::string GetFile() const { return m_Filepath; }
private:
	int m_Width, m_Height, m_Channels;
	std::string m_Filepath;
	unsigned char* m_Data;

	bool str_ends_in(std::string source, std::string suffix);
};