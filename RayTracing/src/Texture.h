#pragma once

#include <string>
#include "Image.h"

class Texture {
public:
	Texture(std::string filepath): m_Filepath(filepath) {
		m_Img = Image3(0, 0, filepath);
	}
	std::string m_Filepath;
	Image3 m_Img;
};
