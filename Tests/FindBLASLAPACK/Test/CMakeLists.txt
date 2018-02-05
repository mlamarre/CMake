cmake_minimum_required(VERSION 3.10)
project(TestFindBLASLAPACK CXX)
include(CTest)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(test_blas_lapack_var main.cxx)
target_link_libraries(test_blas_lapack_var ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
add_test(NAME test_blas_lapack_var COMMAND test_blas_lapack_var)
