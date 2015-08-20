#include <gtest/gtest.h>
#include <matrix_utils.h>

cv::Mat A =
		(cv::Mat_<float>(4, 4) << 3, 0, 2, 0, 2, 0, -2, 0, 0, 1, 1, 0, 0, 0, 0, 1);

cv::Mat Ainv =
		(cv::Mat_<float>(4, 4) << 0.2, 0.2, 0, 0, -0.2, 0.3, 1, 0, 0.2, -0.3, 0, 0, 0, 0, 0, 1);

/**
 * Test for the successful case.
 */
TEST(Invert, SuccessfulInversion) {
	cv::Mat m = invSubmatrix(A, 0, 0, 3, 3);
	ASSERT_TRUE(cv::norm(m, Ainv, cv::NORM_L2) < 10e-4);
}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
