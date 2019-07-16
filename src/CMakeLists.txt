add_library(accu OBJECT accumulator.cpp)
target_include_directories(accu PRIVATE ${Boost_INCLUDE_DIR})
