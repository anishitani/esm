/*
 * matrix_utils.h
 *
 *  Created on: Feb 27, 2015
 *      Author: anishitani
 */

#ifndef MATRIX_UTILS_H_
#define MATRIX_UTILS_H_

#include <fstream>

#include <opencv2/opencv.hpp>

/**
 * Public method for inverting a submatrix.
 * @param m Matrix
 * @param init_x Starting X point
 * @param init_y Starting Y point
 * @param w Width
 * @param h Height
 * @return Matrix with submatrix inverted
 */
cv::Mat invSubmatrix(cv::Mat m, int init_x, int init_y, int w, int h);

/**
 * Public method for writting OpenCV matrices of type T.
 *
 * @param T Matrix data type
 * @param ofilename Output file name
 * @param inputMatrix Input matrix
 */
void writeMatrix(std::string ofilename, cv::InputArray inputMatrix);

#endif /* MATRIX_UTILS_H_ */
