# I Wrote a SIMD Compiler in 12K Lines of Rust

*And then an LLM optimized the kernels better than I could.*

---

## The Elevator Pitch

Eä is a compiler for SIMD kernels. You write a small `.ea` file, run one command, and call it from Python like a normal function. But it runs at native vectorized speed.

```python
import ea
kernel = ea.load("fma.ea")
result = kernel.fma_f32x8(a, b, c, out)  # 6.6× faster than NumPy
```

No ctypes. No header files. No build system. The compiler generates the shared library *and* the Python wrapper. Also Rust, C++, PyTorch, and CMake bindings. It targets x86-64 (AVX2/AVX-512) and AArch64 (NEON).

The whole compiler is 12,000 lines of Rust. 475 tests. One person. I'm not a compiler engineer. But I had a very specific problem, and it turns out that's enough.

## Why

I had a problem that kept repeating. I'd write something in Python, profile it, find a hot loop, and think: "this needs to be fast." And I knew that fast meant C. Fast meant SIMD. I don't have deep experience with either, but I knew that's where the performance lives.

So I'd fumble through some C code (with an LLM helping me write it, which honestly made it worse because now the code worked but I didn't fully understand *why*), fight with ctypes, spend an afternoon debugging pointer arithmetic that the AI got right the first time but I broke while "cleaning up," and eventually get a 5× speedup. Then next week, different project, same dance.

I didn't mind the hard part. Figuring out *what* the kernel should do, thinking about memory access patterns, deciding on vector widths. That's the interesting problem. What I minded was the plumbing. The header files. The build system. The ctypes declarations. The dtype validation. All of it boilerplate, all of it error-prone, none of it the actual work.

So I thought: what if a compiler could handle the plumbing? You write the kernel in a simple language, something that looks like the pseudocode you'd sketch on a whiteboard, and the compiler handles everything else. Compile to a shared library. Auto-generate the Python wrapper. One command. No Makefile.

**The "no glue code" part turned out to be the product.** Not the SIMD. Not the compiler. The fact that you go from `.ea` file to working Python function in one command, with types checked, lengths inferred, and output buffers allocated. That's what makes people actually use it instead of just reading about it.

I didn't know how to build a compiler. But I had the idea, and I wanted to see if it would work.

## The First Attempt (10K Lines of Pain)

I should tell you about the compiler I wrote before this one.

It also targeted LLVM. It also generated SIMD code. The codegen was 10,000 lines in a single file. The parser was hand-written but handled way too many features. There were no hard limits on file size, no style rules, no test discipline. I kept adding things (generics, a module system, type inference) because why not?

The codebase became unmaintainable in about three months. I couldn't change anything without breaking something else. The codegen file had functions that called functions that called functions eight levels deep, and half of them were handling edge cases for features nobody used.

I threw it away.

The lesson: a SIMD kernel compiler doesn't need generics. It doesn't need modules. It doesn't need type inference. It needs to compile `load`, `store`, `fma`, and `splat` correctly, generate clean bindings, and stay small enough that one person can hold it in their head.

## The Hard Rules

For the second attempt, I wrote the rules before I wrote the code:

1. **No file exceeds 500 lines.** Split before you hit the limit.
2. **Every feature proven by end-to-end test.** If it's not tested, it doesn't exist.
3. **No fake functions.** If hardware doesn't support an operation, the compiler errors. No silent fallbacks.
4. **No premature features.** Don't build what isn't needed yet.
5. **Delete, don't comment.** Dead code gets removed.

The 500-line rule was the hardest to follow and the most valuable. It forced me to split the type checker into 7 files, the codegen into 10 files, and the parser into 4 files. Each file does one thing. When I need to change how `store` is type-checked, I open `intrinsics_memory.rs` (309 lines) and the answer is right there. No grepping through 10K lines of spaghetti.

Was it frustrating? Constantly. I'd be in the middle of adding a feature, hit 480 lines, and have to stop and refactor before I could finish. But every time I did, the code got better. The refactor always revealed something. A responsibility that should have been split earlier. A function that was doing two things.

The "no premature features" rule was the other hard one. I kept wanting to add generics. Or a module system. Or traits. And every time, I'd ask myself: does this serve the goal of compiling SIMD kernels to callable shared libraries? The answer was always no. Eä is monomorphic by design. You write `f32x8`, you get `f32x8`. No hidden specialization, no surprise codegen, no combinatorial explosion of type instances.

It's not a limitation. It's the point.

## The Architecture

```
.ea → Lexer (logos) → Parser → Desugar → Type Check → Codegen (LLVM 18) → .o / .so
                                                                          → .ea.json → ea bind
```

The most important insight: **the desugarer is the most important pass.**

Eä has a `kernel` construct that looks like this:

```
export kernel vscale(data: *f32, out: *mut f32, factor: f32)
    over i in n step 8
    tail scalar { out[i] = data[i] * factor }
{
    store(out, i, load(data, i) .* splat(factor))
}
```

The desugarer turns this into a plain function with a while-loop and a tail loop. After desugaring, there are no kernels in the AST. Just functions with loops. The type checker and codegen never see `kernel`. They only see `func`.

This means every downstream pass is simpler. The type checker doesn't need special kernel logic. The codegen doesn't need to handle iteration. The desugarer handles all of it: injecting the `n` parameter, generating the loop variable, building the main loop with step, building the tail loop with the chosen strategy.

The desugar pass is 340 lines. It eliminates an entire class of complexity from the remaining 11,660 lines.

## The Binding Generators

This is the part people don't expect. The compiler generates `.ea.json` metadata describing each exported function's signature. Then `ea bind` reads the JSON and generates idiomatic wrappers:

```bash
ea bind kernel.ea --python --rust --cpp --pytorch --cmake
```

The Python generator does something clever: **length collapsing**. If your kernel takes `(data: *f32, n: i32)`, the generated Python function takes just `data` and fills `n` from `data.size` automatically. Output parameters marked with `out` get auto-allocated. The generated code checks dtypes, casts pointers, and handles all the ctypes plumbing that used to take me an afternoon.

Five binding generators, each 200-460 lines. No serde. The JSON parser is hand-written (65 lines). The generated code is clean enough that you can read it, modify it, and learn from it.

## Error Messages (The Quiet Win)

I spent more time on error messages than on codegen.

The compiler has "did you mean?" suggestions with Levenshtein distance, dot-operator hints ("cannot use '+' on vectors, use '.+'"), `let mut` suggestions, type conversion hints, multi-character underlines showing the full expression span, and clear messages that never leak internal compiler state.

```
kernel.ea:5:12  error[type]: cannot use '+' on vectors. Use '.+' for element-wise vector operations
        return a + b
               ^^^^^
```

Nobody notices good error messages. But everyone notices bad ones. The difference between a user who gives up and a user who fixes their code is often one helpful error message.

## The Autoresearch System (The Fun Part)

This is where it gets interesting.

Inspired by Andrej Karpathy's autoresearch concept, I built an automated optimization loop: an LLM reads the kernel source, the benchmark results, and the history of what's been tried, then proposes a modified kernel. The system compiles it, benchmarks it across multiple data sizes (to catch cache-fitting illusions), verifies correctness against a C reference, and accepts or rejects the change. Then it iterates.

This is where I had the most fun building Eä.

The first time I ran it on the FMA kernel, it found a 10% improvement in 30 iterations. I thought the kernel was already as good as it could get. The LLM found that 12× unrolling with stream stores beat 4× unrolling with regular stores at DRAM scale. I wouldn't have tried that. It sounds like overkill.

Then I let it run on the matrix multiplication kernel. 56% improvement. It switched from ijk to ikj loop order with 8× k-unrolling. I've heard of loop tiling. I couldn't have told you when to apply it. The LLM didn't need to "know." It just tried it and the benchmark said yes.

The thing that surprised me most: you think you have an optimal kernel. You let the LLM iterate 5 times and it finds 20% improvement. Okay, fine, maybe it wasn't optimal. So you let it iterate 50 times on the already-improved kernel. And it *still* finds improvements. The search space for kernel optimization is bigger than your intuition.

27 benchmark kernels, all scored on largest-size (real-world) data with GB/s bandwidth metrics. The system includes bottleneck classification that tells the LLM whether a kernel is DRAM-bound (don't bother with compute tricks), compute-bound (try wider SIMD, more accumulators), or mixed. The biggest wins:

| Kernel | Improvement | What Changed |
|--------|-------------|-------------|
| Bitonic sort | **97%** | Replaced O(n²) Shellsort with sorting network |
| Matmul | **56%** | k×8 unroll, cache-friendly access |
| Conv2d 3×3 | **47%** | 4× column unroll, prefetch, restrict |
| Edge detect | **41%** | f32x4 to f32x8 upgrade |

The humbling part: I'm not an optimization expert. But it turns out you don't need to be one. You need a benchmark harness, a correctness check, and a system that's willing to try things you wouldn't think of.

## The Numbers

```
Source:          12,000 lines of Rust
Tests:           475 end-to-end
Test method:     compile Eä → link with C → run binary → compare stdout
CI:              x86-64, AArch64 (native), Windows
LLVM backend:    18.1 via inkwell 0.8
Binding targets: Python, Rust, C++, PyTorch, CMake
```

Performance on a real workload (16M float32 elements):

```
FMA (fused multiply-add):  6.6× faster than NumPy    37.0 GB/s
Dot product:               matches BLAS               36.6 GB/s
SAXPY:                     2.1× faster than NumPy     35.2 GB/s
```

That's ~37 GB/s on a system with ~40 GB/s memory bandwidth. Near the hardware limit. There's not much room left, which means the code is doing roughly as little unnecessary work as possible.

## What I'd Do Differently

**Start with the binding generator.** I built the compiler first and added bindings later. But the bindings are what make Eä useful. If I'd started by designing the ideal Python API and worked backward to the compiler, some early decisions would have been different.

**Add `ea inspect` earlier.** The instruction analysis tool that shows you vector/scalar instruction counts, FMA operations, load/store ratio, and performance hints. I added it late, but it would have caught optimization issues months earlier.

**Write fewer features, sooner.** Eä has `kernel`, `foreach`, `for`, `while`, structs, output annotations, conditional compilation, static assertions, and 30+ intrinsics. Most users need `kernel`, `load`, `store`, `fma`, and `splat`. I should have shipped a useful subset earlier and iterated based on real usage.

## The Real Story

I'm not a compiler engineer. I don't have a CS degree. I'm the kind of person who has ideas and wants to see if they work.

What changed is the tooling. I built Eä with the help of AI models. Claude for the heavy lifting, my own judgment for the architecture and design decisions. The hard rules came from me (learned the painful way from the first attempt). The implementation speed came from having a capable coding assistant.

A year ago, this would have required a team. Now it's 12,000 lines and one person.

The interesting question isn't "can you build this?" anymore. It's what are you going to build next?

My advice: if you have an idea that feels too ambitious for one person, the calculus has changed. Try it. Set hard rules so the codebase stays manageable. Write tests for everything. And don't be afraid to throw away your first attempt.

I threw away 10K lines of bad compiler and started over. Best decision I made.

---

*Eä is open-source under Apache 2.0. [GitHub](https://github.com/petlukk/eacompute) · [Documentation](https://petlukk.github.io/eacompute/) · [pip install ea-compiler](https://pypi.org/project/ea-compiler/)*
