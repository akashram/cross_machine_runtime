function(set_project_warnings target)
  target_compile_options(${target} PRIVATE
    -Wall                  # standard warnings
    -Wextra                # extra warnings not covered by -Wall
    -Wpedantic             # enforce strict ISO C++ compliance
    -Wshadow               # local variable shadows outer scope variable
    -Wnon-virtual-dtor     # base class with virtual functions but non-virtual destructor
    -Wold-style-cast       # C-style casts (use static_cast/reinterpret_cast explicitly)
    -Wcast-align           # pointer cast increases alignment requirement
    -Woverloaded-virtual   # derived class hides base class virtual function
    -Wconversion           # implicit conversions that may lose data
    -Wsign-conversion      # implicit signed/unsigned conversions
    -Wnull-dereference     # paths that may dereference a null pointer
    -Wdouble-promotion     # float implicitly promoted to double
    -Wformat=2             # printf/scanf format string vulnerabilities
  )
endfunction()
