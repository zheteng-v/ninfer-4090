add_library(ninfer_jinja STATIC
  jinja/lexer.cpp
  jinja/parser.cpp
  jinja/runtime.cpp
  jinja/string.cpp
  jinja/value.cpp
  support/unicode.cpp
)
target_include_directories(ninfer_jinja
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)
