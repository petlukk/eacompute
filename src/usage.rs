pub fn print_usage() {
    eprint!(
        "\
Usage: ea <file.ea> [options]
       ea bind <file.ea> --python [--rust] [--pytorch] [--cmake] [--cpp]
       ea inspect <file.ea> [--avx512] [--target=CPU]

Options:
  -o <name>          Compile and link to executable
  --lib              Produce shared library (.so/.dll) + JSON metadata
  --opt-level=N      Optimization level 0-3 (default: 3)
  --target=CPU       Target CPU (default: native)
  --target-triple=T  Cross-compile target (e.g. aarch64-unknown-linux-gnu)
  --avx512           Enable AVX-512 (f32x16)
  --emit-llvm        Print LLVM IR
  --emit-asm         Emit assembly (.s file)
  --header           Generate C header (.h)
  --emit-ast/--emit-tokens  Print AST or lexer tokens
  --print-target     Print resolved native CPU name and exit
  --help, -h / --version, -V

Subcommands:
  inspect <file.ea>        Analyze kernel: instruction mix, loops, registers
  bind <file.ea> --python  Generate Python/NumPy bindings
  bind <file.ea> --rust    Generate Rust FFI + safe wrappers
  bind <file.ea> --pytorch Generate PyTorch autograd wrappers
  bind <file.ea> --cmake   Generate CMakeLists.txt + EaCompiler.cmake
  bind <file.ea> --cpp     Generate C++ header with std::span
"
    );
}
