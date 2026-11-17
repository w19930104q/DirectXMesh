# Regression tests for DirectXMesh. Included by the root CMakeLists.txt when
# BUILD_TESTING is on (Windows desktop builds).

add_executable(vertexcache-missrate VertexCacheMissRateTest.cpp)
target_link_libraries(vertexcache-missrate PRIVATE ${PROJECT_NAME})
add_test(NAME vertexcache-missrate COMMAND vertexcache-missrate)
