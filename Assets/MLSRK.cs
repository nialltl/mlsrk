// implementation of 'A Moving Least Square Reproducing Kernel Particle Method for Unified Multiphase Continuum Simulation'
// by Xiao-Song Chen, Chen-Feng Li, Geng-Chen Cao, Yun-Tao Jiang & Shi-Min Hu
// based largely on their open source minimal C implementation on acm (see 'Supplemental material') - https://dl.acm.org/doi/abs/10.1145/3414685.3417809

using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;
using UnityEngine;
using UnityEngine.Jobs;
using System.Collections.Generic;
using static Unity.Mathematics.math;

public struct Particle {
    public float2 x;
    public float2 v;
    public float2 f;
    public float mass;
    public float volume;
    public int num_neighbours;
    public float2x2 F; // deformation gradient
    public float3 b; // used for polynomial interpolation (equation 5)
    public float3 dbx; // dbx, dby are derivatives of above
    public float3 dby; 
    public float2 u_h; // see mlsrk paper explanation below equation 12 - u_h is the interpolated velocity at x_i
    public float2x2 σ; // cauchy stress 
}

public class MLSRK : MonoBehaviour {
    #region Setup, kernel functions and utility functions
    // sim bounds
    const int grid_res = 64;
    const float soft_wall_min = 3;
    const float soft_wall_max = grid_res - soft_wall_min;
    
    // parallelisation stuff
    const int division = 32;
    const int max_neighbours = 64;

    // simulation parameters
    const float dt = 0.1f; // timestep
    const int iterations = (int)(1.0f / dt);
    readonly static float2 f_ext = float2(0, 1 * -0.01f);

    const float a = 1.5f; // this is like 'smoothing length' from SPH
    
    // elasticity params for NeoHookean
    const float E = 2.0f;          // Young's Modulus
    const float nu = 0.4f;         // Poisson ratio
    // Lamé parameters get precomputed from the 2 above
    const float mu = E / (2 * (1 + nu));
    const float lambda = E * nu / ((1 + nu) * (1 - 2 * nu));

    NativeArray<Particle> ps; // particles
    NativeArray<int> neighbour_idxs; // acceleration structure

    // interactivity
    const float mouse_spring = 0.3f;
    const float mouse_damp = 0.2f;
    float2 mouse_force;
    SimRenderer sim_renderer;
    int num_particles;
    int selected_idx = -1;

    // utility functions
    static float2x2 outer_product(float2 a, float2 b) {
        return float2x2(a * b.x, a * b.y);
    }
    static float3x3 outer_product(float3 a, float3 b) {
        return float3x3(a * b.x, a * b.y, a * b.z);
    }

    // all the shape functions / spline stuff, and their derivatives. the original paper implementation used cubic splines,
    // but i've found quadratic splines to be sufficient + fast

    static float quadratic_spline(float x) {
        var t = abs(x);
        if (t < 0.5f) return 0.75f - t * t;
        if (t < 1.5f) return 0.5f * pow(1.5f - t, 2);
        return 0;
    }
    static float d_quadratic_spline(float x) {
        var t = abs(x);
        if (t < 0.5f) t = -2 * t;
        else if (t < 1.5f) t = -(1.5f - t);
        else return 0;
        return x > 0 ? t : -t;
    }
    static float Φ(float2 x) {
        return quadratic_spline(x.x) * quadratic_spline(x.y);
    }
    static float2 dΦ(float2 x) {
        return float2(d_quadratic_spline(x.x) * quadratic_spline(x.y), quadratic_spline(x.x) * d_quadratic_spline(x.y));
    }
    #endregion

    // main simulation loop
    void Simulate() {
        // interactivity
        mouse_force = 0;
        var max_dist = float.MaxValue;
        var mp = (Camera.main.ScreenToWorldPoint(Input.mousePosition)) * 10.0f + new Vector3(32, 32, 0);
        var mouse_pos = float2(mp.x, mp.y);
        if (Input.GetMouseButtonDown(0)) {
            for (int i = 0; i < num_particles; i++) {
                var dist = lengthsq(ps[i].x - mouse_pos);
                if (dist < max_dist) {
                    max_dist = dist;
                    selected_idx = i;
                }
            }
        }
        if (Input.GetMouseButton(0)) {
            var pi = ps[selected_idx];
            var delta = mouse_pos - pi.x;
            mouse_force = delta * mouse_spring - pi.v * mouse_damp;
        }

        // main MLSRK jobs
        new GetCorrectionCoefficient() { ps = ps, neighbour_idxs = neighbour_idxs }.Schedule(num_particles, division).Complete();
        // you could split this job up much more for speed and do some sections in parallel, i've just kept it serial for ease of reading
        new MLSRKSim() { ps = ps, neighbour_idxs = neighbour_idxs, mouse_force = mouse_force, selected_idx = selected_idx }.Schedule().Complete();
    }

    #region MLSRK Jobs
    [BurstCompile]
    struct GetCorrectionCoefficient : IJobParallelFor {
        /* Find neighborhood particles inside kernel support range;
           calculate MLSRK correction coefficient b and its gradient for
           the particle neighborhood from Eqn. (8); */

        [NativeDisableParallelForRestriction] public NativeArray<Particle> ps;
        [NativeDisableParallelForRestriction] public NativeArray<int> neighbour_idxs;

        public void Execute(int i) {
            var pi = ps[i];

            float3x3 M = 0;
            float3x3 dMx = 0;
            float3x3 dMy = 0;

            // constant derivatives of basis function
            float3 dhx = float3(0, 1, 0);
            float3 dhy = float3(0, 0, 1);

            int num_neighbours = 0;
            for (int j = 0; j < ps.Length; j++) {
                var pj = ps[j];

                var delta = pj.x - pi.x;
                if (length(delta) < 1.5f * a) {
                    var idx = i * max_neighbours + num_neighbours;
                    neighbour_idxs[idx] = j;

                    // equation 8 for Ni(x_j)
                    var dx = (pj.x - pi.x) / a;
                    // linear basis function
                    float3 h = float3(1, dx.x, dx.y);
                    var Φi = Φ(dx);
                    var gradΦ = dΦ(dx);

                    M += outer_product(h, h) * Φi * pj.volume;

                    // from appendix
                    dMx += (outer_product(dhx, h) * Φi + outer_product(h, dhx) * Φi + outer_product(h, h) * gradΦ.x) * pj.volume;
                    dMy += (outer_product(dhy, h) * Φi + outer_product(h, dhy) * Φi + outer_product(h, h) * gradΦ.y) * pj.volume;

                    ++num_neighbours;
                }
            }
            pi.num_neighbours = num_neighbours;

            dMx /= -a;
            dMy /= -a;

            // regularization from the paper stabilises simulation and ensures matrices can be safely inverted etc, it's like adding an epsilon
            const float regularization = 0.001f;
            var Mr = pow(a, 2) * float3x3(
                0, 0, 0,
                0, regularization, 0,
                0, 0, regularization
            );
            M += Mr;

            var M_inv = inverse(M);

            // linear basis function at 0
            float3 h0 = float3(1, 0, 0);
            var b = mul(h0, M_inv);

            var dMx_inv = mul(-M_inv, mul(dMx, M_inv));
            var dMy_inv = mul(-M_inv, mul(dMy, M_inv));

            var dbx = mul(h0, dMx_inv);
            var dby = mul(h0, dMy_inv);

            pi.dbx = dbx;
            pi.dby = dby;

            pi.b = b;
            ps[i] = pi;
        }
    }
    
    [BurstCompile]
    struct MLSRKSim : IJob {
        /* calculate velocity gradient ∇u from Eqn. (18), update
            deformation gradient F_n+1 = (I + (∇u_n+1 )^T Δt)F_n
            following Eqn. (17); */

        [NativeDisableParallelForRestriction] public NativeArray<Particle> ps;
        [NativeDisableParallelForRestriction] public NativeArray<int> neighbour_idxs;
        
        public int selected_idx;
        public float2 mouse_force;

        public void Execute() {
            // calculate stress per-particle
            for (int i = 0; i < ps.Length; i++) {
                var pi = ps[i];

                var F = pi.F;
                var J = determinant(F);
                var FFT = mul(F, transpose(F));
                var I = float2x2(1, 0, 0, 1);

                // eq. 22
                var σ = (1.0f / J) * (mu * (FFT - I) + lambda * log(J) * I);
                pi.σ = σ;

                ps[i] = pi;
            }

            // force calculation based on stress, individual particle raw velocity
            for (int i = 0; i < ps.Length; i++) {
                var pi = ps[i];

                float2 f = 0;
                float m = 0;
                
                for (int k = 0; k < pi.num_neighbours; ++k) {
                    var idx_into_neighbours = i * max_neighbours + k;
                    var j = neighbour_idxs[idx_into_neighbours];
                    var pj = ps[j];

                    var b = pj.b;
                    var σj = pj.σ;

                    var dbx = pj.dbx;
                    var dby = pj.dby;

                    float2 dx = (pi.x - pj.x) / a;
                    float3 h = float3(1, dx.x, dx.y);

                    // shape function for pi at neighbour sample point j
                    float N = dot(b, h) * Φ(dx) * pi.volume;
                    float2 dN = ((float2(dot(dbx, h), dot(dby, h)) - float2(b[1], b[2]) / a) * Φ(dx) - dot(b, h) * dΦ(dx) / a) * pi.volume;

                    f += -mul(dN, σj * pj.volume);
                    var rho_j = pj.mass / pj.volume;
                    m += N * rho_j * pj.volume;
                }

                // interactivity
                var g = f_ext;
                if (i == selected_idx) {
                    g += mouse_force;
                }
                pi.v += (f / m + g) * dt;

                // boundary conditions - simple soft hookean springs
                const float wall_spring = 0.5f;
                const float wall_damp = 0.2f;
                if (pi.x.x < soft_wall_min) {
                    pi.v.x += (soft_wall_min - pi.x.x) * wall_spring - pi.v.x * wall_damp;
                }
                if (pi.x.x > soft_wall_max) {
                    pi.v.x += (soft_wall_max - pi.x.x) * wall_spring - pi.v.x * wall_damp;
                }
                if (pi.x.y < soft_wall_min) {
                    pi.v.y += (soft_wall_min - pi.x.y) * wall_spring - pi.v.y * wall_damp;
                    pi.v.x = 0;
                }
                if (pi.x.y > soft_wall_max) {
                    pi.v.y += (soft_wall_max - pi.x.y) * wall_spring - pi.v.y * wall_damp;
                }

                ps[i] = pi;
            }

            // update deformation gradient & get interpolated velocity
            for (int i = 0; i < ps.Length; i++) {
                var pi = ps[i];

                var b = pi.b;

                var dbx = pi.dbx;
                var dby = pi.dby;

                float2x2 du = 0;
                float2 u_h = 0;

                for (int k = 0; k < pi.num_neighbours; ++k) {
                    var idx_into_neighbours = i * max_neighbours + k;
                    var j = neighbour_idxs[idx_into_neighbours];
                    var pj = ps[j];

                    float2 dx = (pj.x - pi.x) / a;
                    float3 h = float3(1, dx.x, dx.y);

                    // shape function for pi at neighbour sample point j
                    float N = dot(b, h) * Φ(dx) * pj.volume;
                    float2 dN = ((float2(dot(dbx, h), dot(dby, h)) - float2(b[1], b[2]) / a) * Φ(dx) - dot(b, h) * dΦ(dx) / a) * pj.volume;

                    // equation 8 for Ni(x_j)
                    du += outer_product(dN, pj.v);
                    
                    // calculating u_h(xi) for equation 12 velocity blending
                    u_h += N * pj.v;
                }

                pi.u_h = u_h;

                // update deformation gradient
                pi.F += mul(transpose(du), pi.F) * dt;

                ps[i] = pi;
            }
            
            // integrate position forward
            for (int i = 0; i < ps.Length; i++) {
                var pi = ps[i];

                var u_h = pi.u_h;

                // update position using eq 12 - this is like PIC/FLIP blending,
                // where pi.v is the raw particle velocity and u_h is the 'grid' velocity - obvs there's no grid but it's like a smoothed node position
                const float blend_alpha = 0.95f;
                var u_tilde = blend_alpha * pi.v + (1.0f - blend_alpha) * u_h;
                pi.x += u_h * dt;
                pi.v = u_tilde;

                // update per-particle volume. this is a simplified version of the C implementation, it's not great - will gain volume with large deformations
                // see the text below equation 19 in the paper for a corrected volume / deformation gradient update scheme, more important for fluids and granular materials
                const float vol_0 = 1.0f;
                var vol = determinant(pi.F) * vol_0;
                pi.volume = vol;

                ps[i] = pi;
            }
        }
    }
    #endregion

    #region Initialisation and Unity boilerplate
    void Start() {
        // initialise particles
        var temp_positions = new List<float2>();
        const float box_x = grid_res / 2.0f, box_y = grid_res / 2.0f;
        const float box_width = 8, box_height = 8;
        var spacing = a * 0.5f;
        for (var i = -box_width; i <= box_width; i += spacing) {
            for (var j = -box_height; j <= box_height; j += spacing) {
                var pos = float2(i, j) + float2(box_x, box_y);
                temp_positions.Add(pos);
            }
        }
        num_particles = temp_positions.Count;
        ps = new NativeArray<Particle>(num_particles, Allocator.Persistent);

        // initialise neighbourhood lookup list
        neighbour_idxs = new NativeArray<int>(num_particles * max_neighbours, Allocator.Persistent);

        // initialise particles
        for (int i = 0; i < num_particles; ++i) {
            var pi = new Particle();
            pi.x = temp_positions[i];
            pi.v = 0;
            pi.f = 0;
            pi.mass = 1.0f;
            pi.volume = 1.0f;
            pi.F = float2x2(1, 0, 0, 1);
            ps[i] = pi;
        }

        // boilerplate rendering code handled elsewhere
        sim_renderer = GameObject.FindObjectOfType<SimRenderer>();
        sim_renderer.Initialise(num_particles, System.Runtime.InteropServices.Marshal.SizeOf(new Particle()));
    }

    private void Update() {
        for (int i = 0; i < iterations; ++i) {
            Simulate();
        }

        // in case you wanna debug anything - here's some cpu DrawLine rendering in-editor
        /*var colour = new Color(235 / 255.0f, 152 / 255.0f, 52 / 255.0f, 0.1f);
        for (int i = 0; i < num_particles; i++) {
            var pi = ps[i];
            var sp = (new Vector2(pi.x.x, pi.x.y) - Vector2.one * (grid_res / 2)) * 0.1f;
            for (int k = 0; k < pi.num_neighbours; ++k) {
                var idx_into_neighbours = i * max_neighbours + k;
                var j = neighbour_idxs[idx_into_neighbours];
                var pj = ps[j];

                var sq = (new Vector2(pj.x.x, pj.x.y) - Vector2.one * (grid_res / 2)) * 0.1f;

                Debug.DrawLine(sp, sq, colour);
            }
        }*/

        sim_renderer.RenderFrame(ps);
    }

    private void OnDestroy() {
        ps.Dispose();
        neighbour_idxs.Dispose();
    }
    #endregion
}

