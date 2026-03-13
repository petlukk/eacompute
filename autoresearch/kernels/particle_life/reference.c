#include <math.h>

void particle_life_step_ref(
    float *px, float *py,
    float *vx, float *vy,
    const int *types,
    const float *matrix,
    int n, int num_types,
    float r_max, float dt, float friction, float size
) {
    float r_max2 = r_max * r_max;
    for (int i = 0; i < n; i++) {
        float xi = px[i];
        float yi = py[i];
        int ti = types[i];
        float fx = 0.0f;
        float fy = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = px[j] - xi;
            float dy = py[j] - yi;
            float dist2 = dx * dx + dy * dy;
            if (dist2 > 0.0f && dist2 < r_max2) {
                float dist = sqrtf(dist2);
                float strength = matrix[ti * num_types + types[j]];
                float force = strength * (1.0f - dist / r_max);
                fx += force * dx / dist;
                fy += force * dy / dist;
            }
        }

        vx[i] = (vx[i] + fx * dt) * friction;
        vy[i] = (vy[i] + fy * dt) * friction;
        px[i] = px[i] + vx[i];
        py[i] = py[i] + vy[i];

        float cur_px = px[i];
        float cur_py = py[i];
        if (cur_px < 0.0f) px[i] = cur_px + size;
        if (cur_px >= size) px[i] = cur_px - size;
        if (cur_py < 0.0f) py[i] = cur_py + size;
        if (cur_py >= size) py[i] = cur_py - size;
    }
}
