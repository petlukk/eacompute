# Structs

Structs in Ea are plain data containers with C-compatible memory layout. They have no methods, no constructors, no impl blocks. They exist so you can pass structured data between Ea kernels and host languages.

## Defining a struct

```
struct Particle {
    x: f32,
    y: f32,
    mass: f32,
}
```

Fields can be any scalar type. The memory layout matches C struct layout, so a `Particle` in Ea is identical to:

```c
typedef struct { float x; float y; float mass; } Particle;
```

## Creating and accessing

Inside Ea functions, you create a struct with literal syntax and access fields with dot notation:

```
func main() {
    let p: Particle = Particle { x: 1.0, y: 2.0, mass: 10.0 }
    println(p.x)      // 1
    println(p.mass)    // 10
}
```

Mutable structs support field assignment:

```
func main() {
    let mut p: Particle = Particle { x: 0.0, y: 0.0, mass: 1.0 }
    p.x = 3.5
    p.y = 7.0
    println(p.x)    // 3.5
}
```

## Struct pointers

In exported functions, structs are typically passed via pointer from the host language:

```
struct Point {
    x: f32,
    y: f32,
}

export func get_x(p: *Point) -> f32 {
    return p.x
}

export func set_point(p: *mut Point, nx: f32, ny: f32) {
    p.x = nx
    p.y = ny
}
```

Read-only pointers (`*Point`) allow field reads. Mutable pointers (`*mut Point`) allow field writes.

## Arrays of structs

Pointer indexing works with struct arrays. Each element is a full struct:

```
struct Vec2 {
    x: f32,
    y: f32,
}

export func sum_x(vecs: *Vec2, n: i32) -> f32 {
    let mut total: f32 = 0.0
    let mut i: i32 = 0
    while i < n {
        total = total + vecs[i].x
        i = i + 1
    }
    return total
}
```

From C, this is called with a pointer to a contiguous array of structs:

```c
typedef struct { float x; float y; } Vec2;
extern float sum_x(const Vec2*, int);

Vec2 vecs[] = { {1.0f, 10.0f}, {2.0f, 20.0f}, {3.0f, 30.0f} };
float result = sum_x(vecs, 3);  // 6.0
```

## Passing from Python

Since Ea structs match C layout, you can use NumPy structured arrays or ctypes:

```python
import numpy as np

particle_dtype = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('mass', np.float32),
])

particles = np.zeros(1000, dtype=particle_dtype)
```

The generated Python bindings handle the pointer passing automatically.

## Limitations

- No methods or impl blocks. Structs are data only.
- No nested structs.
- No generics. Write separate struct definitions for each concrete type.
- Struct fields must be scalar types.
