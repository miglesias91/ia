
// gtest
#include <gtest/gtest.h>

#ifdef DEBUG | _DEBUG
// vld
#include <vld.h>
#endif

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    std::getchar();

    return result;
}