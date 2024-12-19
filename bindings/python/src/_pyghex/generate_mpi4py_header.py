import sys
import mpi4py

def main():
    if len(sys.argv) != 2:
        print("Usage: generate_mpi4py_header.py <output_header_path>")
        sys.exit(1)

    output_path = sys.argv[1]
    version = mpi4py.__version__

    header_content = f"""\
#pragma once

#include <mpi4py/mpi4py.h>

namespace pyghex
{{
namespace
{{
inline constexpr char const* mpi4py_version = "{version}";
}} // anonymous namespace
}} // namespace pyghex
"""

    with open(output_path, "w") as f:
        f.write(header_content)
    print(f"Header file generated at: {output_path}")

if __name__ == "__main__":
    main()
