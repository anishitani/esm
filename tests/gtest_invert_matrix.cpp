#include <gtest/gtest.h>
#include <matrix_utils.h>

TEST(Empty, Return){
	EXPECT_FALSE(false);
}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
