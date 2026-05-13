use std::process;

pub(crate) fn handle_bind(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: ea bind <file.ea> --python [--rust] [--pytorch] [--cmake] [--cpp]");
        process::exit(1);
    }
    let input_file = &args[0];
    let mut python = false;
    let mut rust = false;
    let mut pytorch = false;
    let mut cmake = false;
    let mut cpp = false;
    for arg in &args[1..] {
        match arg.as_str() {
            "--python" => python = true,
            "--rust" => rust = true,
            "--pytorch" => pytorch = true,
            "--cmake" => cmake = true,
            "--cpp" => cpp = true,
            other => {
                eprintln!("error: unknown bind option '{other}'");
                process::exit(1);
            }
        }
    }
    if !python && !rust && !pytorch && !cmake && !cpp {
        eprintln!(
            "error: ea bind requires at least one of --python, --rust, --pytorch, --cmake, --cpp"
        );
        process::exit(1);
    }

    let stem = std::path::Path::new(input_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    // Read JSON metadata (needed by all generators except --cmake)
    let json_str = if python || rust || pytorch || cpp {
        let json_path = format!("{input_file}.json");
        match std::fs::read_to_string(&json_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot read '{json_path}': {e}");
                eprintln!("hint: compile with --lib first to generate JSON metadata");
                process::exit(1);
            }
        }
    } else {
        String::new()
    };

    if python {
        let code = match ea_compiler::bind_python::generate(&json_str, stem) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to generate Python bindings: {e}");
                process::exit(1);
            }
        };
        let path = format!("{stem}.py");
        write_or_exit(&path, &code);
    }

    if rust {
        let code = match ea_compiler::bind_rust::generate(&json_str, stem) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to generate Rust bindings: {e}");
                process::exit(1);
            }
        };
        let path = format!("{stem}.rs");
        write_or_exit(&path, &code);
    }

    if pytorch {
        let code = match ea_compiler::bind_pytorch::generate(&json_str, stem) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to generate PyTorch bindings: {e}");
                process::exit(1);
            }
        };
        let path = format!("{stem}_torch.py");
        write_or_exit(&path, &code);
    }

    if cmake {
        let (cmakelists, ea_cmake) = ea_compiler::bind_cmake::generate(stem);
        write_or_exit("CMakeLists.txt", &cmakelists);
        write_or_exit("EaCompiler.cmake", &ea_cmake);
    }

    if cpp {
        let code = match ea_compiler::bind_cpp::generate(&json_str, stem) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to generate C++ bindings: {e}");
                process::exit(1);
            }
        };
        let path = format!("{stem}.hpp");
        write_or_exit(&path, &code);
    }
}

fn write_or_exit(path: &str, content: &str) {
    if let Err(e) = std::fs::write(path, content) {
        eprintln!("error: cannot write '{path}': {e}");
        process::exit(1);
    }
    eprintln!("wrote {path}");
}
