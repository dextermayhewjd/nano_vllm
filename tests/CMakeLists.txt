include(FetchContent)

# ===== GoogleTest =====
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)


FetchContent_MakeAvailable(googletest)

# ===== Unit tests =====
add_subdirectory(unit)
