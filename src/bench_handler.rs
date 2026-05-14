use std::process;

pub(crate) fn handle_bench(args: &[String]) {
    #[cfg(feature = "llvm")]
    {
        if args.iter().any(|a| a == "--help" || a == "-h") {
            print_bench_usage();
            return;
        }
        match ea_compiler::bench::run(args) {
            Ok(code) => process::exit(code),
            Err(e) => {
                eprintln!("error: {e}");
                process::exit(1);
            }
        }
    }
    #[cfg(not(feature = "llvm"))]
    {
        let _ = args;
        eprintln!("error: ea bench requires the 'llvm' feature");
        process::exit(1);
    }
}

#[cfg(feature = "llvm")]
fn print_bench_usage() {
    eprintln!(
        "\
Usage: ea bench <manifest.toml> [options]

Options:
  --target=CPU         Forward to the kernel build (default: native)
  --avx512             Forward: enable AVX-512
  --fp16               Forward: enable ARM FEAT_FP16
  --i8mm               Forward: enable ARM I8MM
  --dotprod            Forward: enable ARM dot-product
  --opt-level=N        Forward: optimization level (default: 3)
  --update-baseline    Overwrite manifest.baseline with this run's result
  --no-diff            Skip baseline comparison
  --out PATH           Write result JSON to PATH instead of stdout"
    );
}
