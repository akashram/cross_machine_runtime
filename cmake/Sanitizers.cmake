set(RUNTIME_SANITIZER "" CACHE STRING
  "Sanitizer to enable: address, thread, or undefined")

if(RUNTIME_SANITIZER)
  message(STATUS "Sanitizer: ${RUNTIME_SANITIZER}")
  add_compile_options(-fsanitize=${RUNTIME_SANITIZER} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${RUNTIME_SANITIZER})
endif()
