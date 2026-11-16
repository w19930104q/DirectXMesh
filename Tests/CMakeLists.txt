# Regression tests for DirectXMesh. Included by the root CMakeLists.txt when
# BUILD_TESTING is on (Windows desktop builds).

add_executable(optimizelru-degenerate OptimizeLRUDegenerateTest.cpp)
target_link_libraries(optimizelru-degenerate PRIVATE ${PROJECT_NAME})
add_test(NAME optimizelru-degenerate COMMAND optimizelru-degenerate)
