add_test(NAME FindBLASLAPACK.Test COMMAND
  ${CMAKE_CTEST_COMMAND} -C $<CONFIGURATION>
  --build-and-test
  "${CMake_SOURCE_DIR}/Tests/FindBLASLAPAC/Test"
  "${CMake_BINARY_DIR}/Tests/FindBLASLAPAC/Test"
  ${build_generator_args}
  --build-project TestFindBLASLAPACK
  --build-options ${build_options}
  --test-command ${CMAKE_CTEST_COMMAND} -V -C $<CONFIGURATION>
  )