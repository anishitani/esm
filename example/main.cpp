#include <cstdio>

#include <opencv2/opencv.hpp>

double degree_to_rad(double degree) {
	const double halfC = CV_PI / 180;
	return degree * halfC;
}

void rotate_around_center(cv::Mat image, double angle) {
	int w = image.cols;
	int h = image.rows;

	cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(w/2,h/2),10,1);

	cv::warpAffine(image, image, rot, image.size());
}

int main(int argc, char **argv) {

	std::string win("Image");

	cv::Mat image = cv::imread("blocks/anishitani/esm/example/baboon.jpg",
			CV_LOAD_IMAGE_GRAYSCALE);

	// Fails on empty image
	if (image.empty()) {
		fprintf(stderr, "Couldn't open image!\n");
		return -1;
	}

	int w = image.cols;
	int h = image.rows;

	// Resize image
	cv::Matx23f affine(0.5, 0, w / 4, 0, 0.5, h / 4);
	cv::warpAffine(image, image, affine, cv::Size_<int>(h, w));

	double angle = degree_to_rad(10);
	cv::Mat warped(image);

	rotate_around_center(warped, angle);

	cv::imshow(win, warped);
	cv::waitKey();

	return 0;
}
