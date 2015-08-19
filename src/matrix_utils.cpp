/*
 * matrix_utils.cpp
 *
 *  Created on: Feb 27, 2015
 *      Author: anishitani
 */

#include "matrix_utils.h"

cv::Mat invSubmatrix(cv::Mat m, int init_x, int init_y, int w, int h) {
	CV_Assert((m.cols < init_x + w) || (m.rows + h));

	cv::Mat _m(m.clone());
	cv::Mat(m(cv::Rect(init_x, init_y, w, h)).inv()).copyTo(_m(cv::Rect(init_x, init_y, w, h)));

	return _m;
}

void writeMatrix(std::string ofilename, cv::InputArray inputMatrix) {
	std::ofstream ofile;
	ofile.open(ofilename.c_str(), std::ofstream::out | std::ofstream::app);
	cv::Mat m = inputMatrix.getMat();
	if (ofile.is_open()) {
		for (int i = 0; i < m.rows * m.cols; i++) {
			ofile << m.at<float>(i) << (((i+1)%m.cols) ? ' ' : '\n');
		}
	}
	ofile.close();
}
