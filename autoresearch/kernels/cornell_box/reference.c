#include <math.h>

typedef struct { float x, y, z; } Vec3;
typedef struct { float t; int id; } HitInfo;

static Vec3 v3(float x, float y, float z) { return (Vec3){x, y, z}; }
static Vec3 v3_add(Vec3 a, Vec3 b) { return (Vec3){a.x+b.x, a.y+b.y, a.z+b.z}; }
static Vec3 v3_sub(Vec3 a, Vec3 b) { return (Vec3){a.x-b.x, a.y-b.y, a.z-b.z}; }
static Vec3 v3_scale(Vec3 v, float s) { return (Vec3){v.x*s, v.y*s, v.z*s}; }
static float v3_dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static float v3_len(Vec3 v) { return sqrtf(v3_dot(v, v)); }

static Vec3 v3_normalize(Vec3 v) {
    float inv = 1.0f / sqrtf(v3_dot(v, v));
    return v3_scale(v, inv);
}

static Vec3 v3_reflect(Vec3 d, Vec3 n) {
    float dn = v3_dot(d, n);
    return v3_sub(d, v3_scale(n, 2.0f * dn));
}

static float hit_sphere(Vec3 ro, Vec3 rd, float cx, float cy, float cz, float r) {
    float ocx = ro.x - cx, ocy = ro.y - cy, ocz = ro.z - cz;
    float a = rd.x*rd.x + rd.y*rd.y + rd.z*rd.z;
    float b = 2.0f * (ocx*rd.x + ocy*rd.y + ocz*rd.z);
    float c = ocx*ocx + ocy*ocy + ocz*ocz - r*r;
    float disc = b*b - 4.0f*a*c;
    if (disc < 0.0f) return -1.0f;
    float sd = sqrtf(disc);
    float inv2a = 1.0f / (2.0f * a);
    float t1 = (-b - sd) * inv2a;
    if (t1 > 0.001f) return t1;
    float t2 = (-b + sd) * inv2a;
    if (t2 > 0.001f) return t2;
    return -1.0f;
}

static HitInfo closest_hit(Vec3 ro, Vec3 rd) {
    float best_t = 99999.0f;
    int best_id = 0;

    float t_floor = -ro.y / rd.y;
    if (t_floor > 0.001f && t_floor < best_t) {
        float hx = ro.x + t_floor * rd.x;
        float hz = ro.z + t_floor * rd.z;
        if (hx >= 0.0f && hx <= 1.0f && hz >= 0.0f && hz <= 1.0f) {
            best_t = t_floor; best_id = 1;
        }
    }

    float t_ceil = (1.0f - ro.y) / rd.y;
    if (t_ceil > 0.001f && t_ceil < best_t) {
        float hx = ro.x + t_ceil * rd.x;
        float hz = ro.z + t_ceil * rd.z;
        if (hx >= 0.0f && hx <= 1.0f && hz >= 0.0f && hz <= 1.0f) {
            best_t = t_ceil; best_id = 2;
        }
    }

    float t_back = (1.0f - ro.z) / rd.z;
    if (t_back > 0.001f && t_back < best_t) {
        float hx = ro.x + t_back * rd.x;
        float hy = ro.y + t_back * rd.y;
        if (hx >= 0.0f && hx <= 1.0f && hy >= 0.0f && hy <= 1.0f) {
            best_t = t_back; best_id = 3;
        }
    }

    float t_left = -ro.x / rd.x;
    if (t_left > 0.001f && t_left < best_t) {
        float hy = ro.y + t_left * rd.y;
        float hz = ro.z + t_left * rd.z;
        if (hy >= 0.0f && hy <= 1.0f && hz >= 0.0f && hz <= 1.0f) {
            best_t = t_left; best_id = 4;
        }
    }

    float t_right = (1.0f - ro.x) / rd.x;
    if (t_right > 0.001f && t_right < best_t) {
        float hy = ro.y + t_right * rd.y;
        float hz = ro.z + t_right * rd.z;
        if (hy >= 0.0f && hy <= 1.0f && hz >= 0.0f && hz <= 1.0f) {
            best_t = t_right; best_id = 5;
        }
    }

    float t_s1 = hit_sphere(ro, rd, 0.65f, 0.25f, 0.55f, 0.25f);
    if (t_s1 > 0.001f && t_s1 < best_t) { best_t = t_s1; best_id = 6; }

    float t_s2 = hit_sphere(ro, rd, 0.3f, 0.18f, 0.35f, 0.18f);
    if (t_s2 > 0.001f && t_s2 < best_t) { best_t = t_s2; best_id = 7; }

    return (HitInfo){best_t, best_id};
}

static Vec3 get_normal(int id, Vec3 pos) {
    if (id == 1) return v3(0,1,0);
    if (id == 2) return v3(0,-1,0);
    if (id == 3) return v3(0,0,-1);
    if (id == 4) return v3(1,0,0);
    if (id == 5) return v3(-1,0,0);
    if (id == 6) return v3_normalize(v3_sub(pos, v3(0.65f,0.25f,0.55f)));
    return v3_normalize(v3_sub(pos, v3(0.3f,0.18f,0.35f)));
}

static Vec3 get_color(int id) {
    if (id == 4) return v3(0.65f, 0.05f, 0.05f);
    if (id == 5) return v3(0.12f, 0.45f, 0.15f);
    return v3(0.73f, 0.73f, 0.73f);
}

static Vec3 shade_point(Vec3 pos, Vec3 normal, Vec3 color) {
    Vec3 light = v3(0.5f, 0.95f, 0.5f);
    Vec3 to_light = v3_sub(light, pos);
    float light_dist = v3_len(to_light);
    Vec3 light_dir = v3_scale(to_light, 1.0f / light_dist);

    float ndotl = v3_dot(normal, light_dir);
    if (ndotl < 0.0f) ndotl = 0.0f;

    Vec3 shadow_origin = v3_add(pos, v3_scale(normal, 0.002f));
    HitInfo shadow_hit = closest_hit(shadow_origin, light_dir);
    float shadow = 1.0f;
    if (shadow_hit.id > 0 && shadow_hit.t < light_dist) shadow = 0.0f;

    float brightness = 0.15f + ndotl * 0.85f * shadow;
    return (Vec3){color.x * brightness, color.y * brightness, color.z * brightness};
}

static Vec3 trace(Vec3 ro, Vec3 rd, int depth) {
    if (depth > 1) return v3(0,0,0);
    HitInfo hit = closest_hit(ro, rd);
    if (hit.id == 0) return v3(0,0,0);
    Vec3 pos = v3_add(ro, v3_scale(rd, hit.t));
    Vec3 normal = get_normal(hit.id, pos);
    if (hit.id == 7) {
        Vec3 refl_dir = v3_reflect(rd, normal);
        Vec3 refl_origin = v3_add(pos, v3_scale(normal, 0.002f));
        return trace(refl_origin, refl_dir, depth + 1);
    }
    Vec3 color = get_color(hit.id);
    return shade_point(pos, normal, color);
}

void render_ref(float *out, int width, int height) {
    Vec3 cam = v3(0.5f, 0.5f, -1.0f);
    float fh = (float)height;
    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            float u = ((float)px + 0.5f - (float)width * 0.5f) / fh;
            float vv = -((float)py + 0.5f - fh * 0.5f) / fh;
            Vec3 rd = v3_normalize(v3(u, vv, 1.0f));
            Vec3 color = trace(cam, rd, 0);
            int idx = (py * width + px) * 3;
            out[idx] = color.x;
            out[idx+1] = color.y;
            out[idx+2] = color.z;
        }
    }
}
