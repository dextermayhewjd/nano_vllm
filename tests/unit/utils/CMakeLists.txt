add_executable(test_utils
  test_status.cc
)
target_link_libraries(test_utils
  PRIVATE
    llm_engine
    gtest_main
)
add_test(NAME utils COMMAND test_utils)