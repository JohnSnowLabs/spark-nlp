cmake_minimum_required(VERSION 3.1)
project(sentencepiece-jni)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(JNI)

include(ExternalProject)

set(sentencepiece_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece/src/sentencepiece-build/src/libsentencepiece.a)
set(sentencepiece_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece/src/sentencepiece/src)
ExternalProject_Add(sentencepiece
    PREFIX sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.1.82
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_BUILD_TYPE:STRING=Release
)

add_library(sentencepiece_jni SHARED com_johnsnowlabs_nlp_sentencepiece_SentencePieceJNI.cc)
add_dependencies(sentencepiece_jni sentencepiece)
target_include_directories(sentencepiece_jni PUBLIC
    ${JNI_INCLUDE_DIRS}
    ${sentencepiece_INCLUDE_DIRS})
target_link_libraries(sentencepiece_jni ${sentencepiece_STATIC_LIBRARIES})

install(TARGETS sentencepiece_jni DESTINATION ${CMAKE_INSTALL_PREFIX})
