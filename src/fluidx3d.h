#ifndef FLUIDX3D_H
#define FLUIDX3D_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ── Opaque solver handle ── */
typedef struct FluidX3D_Solver FluidX3D_Solver;

/* ── Configuration ── */
typedef struct {
	/* Grid */
	uint32_t nx, ny, nz;
	uint32_t dx, dy, dz;             /* 1,1,1 = single-GPU */

	/* Physics */
	float    nu;                     /* kinematic viscosity */
	float    fx, fy, fz;             /* volume force */
	float    sigma;                  /* surface tension (0=disabled) */
	float    alpha, beta;            /* thermal diffusion / expansion */
	uint32_t particles_n;            /* particle count (0=disabled) */
	float    particles_rho;          /* particle density ratio (1.0=passive) */

	/* LBM variant */
	int32_t  velocity_set;           /* 9, 15, 19, or 27 */
	int32_t  collision;              /* 0=SRT, 1=TRT */
	int32_t  precision;              /* 0=FP32, 1=FP16S, 2=FP16C */

	/* Extensions (bitmask of FLUIDX3D_EXT_* flags) */
	uint32_t extensions;

	/* Device selection */
	int32_t  device_count;
	int32_t  device_ids[8];          /* negative = auto-select */

	/* Unit conversion */
	float    si_length, si_velocity, si_density;

	/* Control */
	uint64_t total_steps;            /* UINT64_MAX = infinite */

	/* Benchmark mode */
	bool     benchmark;
} FluidX3D_Config;

/* Extension flags */
#define FLUIDX3D_EXT_VOLUME_FORCE           (1u << 0)
#define FLUIDX3D_EXT_FORCE_FIELD            (1u << 1)
#define FLUIDX3D_EXT_EQUILIBRIUM_BOUNDARIES  (1u << 2)
#define FLUIDX3D_EXT_MOVING_BOUNDARIES      (1u << 3)
#define FLUIDX3D_EXT_SURFACE                (1u << 4)
#define FLUIDX3D_EXT_TEMPERATURE            (1u << 5)
#define FLUIDX3D_EXT_SUBGRID                (1u << 6)
#define FLUIDX3D_EXT_PARTICLES              (1u << 7)

/* Field identifiers */
#define FLUIDX3D_FIELD_RHO       0
#define FLUIDX3D_FIELD_U         1    /* 3-component: x,y,z interleaved */
#define FLUIDX3D_FIELD_FLAGS     2    /* uint8_t per cell */
#define FLUIDX3D_FIELD_FORCE     3    /* 3-component, needs FORCE_FIELD */
#define FLUIDX3D_FIELD_PHI       4    /* scalar, needs SURFACE */
#define FLUIDX3D_FIELD_TEMP      5    /* scalar, needs TEMPERATURE */
#define FLUIDX3D_FIELD_PARTICLES 6    /* 3-component positions, needs PARTICLES */

/* ── Lifecycle ── */

/** Create solver with given configuration. Returns NULL on failure. */
FluidX3D_Solver* fluidx3d_create(const FluidX3D_Config* config);

/** Destroy solver and free all resources. */
void fluidx3d_destroy(FluidX3D_Solver* solver);

/* ── Simulation ── */

/** Initialize without stepping (copies host→device, runs init kernel). */
void fluidx3d_initialize(FluidX3D_Solver* solver);

/** Run N steps. total_steps sets progress denominator (UINT64_MAX=infinite).
 *  Returns actual steps executed. */
uint64_t fluidx3d_run(FluidX3D_Solver* solver, uint64_t steps, uint64_t total_steps);

/** Set volume force at runtime. */
void fluidx3d_set_force(FluidX3D_Solver* solver, float fx, float fy, float fz);

/** Get current time step. */
uint64_t fluidx3d_get_step(const FluidX3D_Solver* solver);

/** Get grid dimensions. */
void fluidx3d_get_dims(const FluidX3D_Solver* solver,
                       uint32_t* nx, uint32_t* ny, uint32_t* nz);

/* ── Field Access ── */

/** Number of scalar elements for a field (1×N for scalar, 3×N for vector). */
uint64_t fluidx3d_field_size(const FluidX3D_Solver* solver, int field_id);

/** Number of components per cell (1=scalar, 3=vector). */
int fluidx3d_field_components(int field_id);

/** Read field from GPU to pre-allocated host buffer. Returns element count.
 *  Buffer must be at least fluidx3d_field_size() elements large.
 *  For vector fields (U, FORCE, PARTICLES), data layout is SoA:
 *    float* x = buf; float* y = buf + N; float* z = buf + 2*N;
 */
uint64_t fluidx3d_read_field(FluidX3D_Solver* solver, int field_id,
                             void* host_buffer, uint64_t buffer_size);

/** Write field from host buffer to GPU (for initial/boundary conditions).
 *  Must be called BEFORE fluidx3d_initialize() or after fluidx3d_reset(). */
void fluidx3d_write_field(FluidX3D_Solver* solver, int field_id,
                          const void* host_buffer, uint64_t buffer_size);

/** Convenience: read field as float array (allocates internally).
 *  Caller must free() the returned pointer. Returns NULL on failure. */
float* fluidx3d_read_field_float(FluidX3D_Solver* solver, int field_id,
                                 uint64_t* out_count);

/** Convenience: read field as uint8_t array (for FLAGS field).
 *  Caller must free() the returned pointer. Returns NULL on failure. */
uint8_t* fluidx3d_read_field_uchar(FluidX3D_Solver* solver, int field_id,
                                   uint64_t* out_count);

/* ── Mesh Loading ── */

/** Voxelize an STL file into the simulation grid. */
void fluidx3d_voxelize_stl(FluidX3D_Solver* solver,
                           const char* stl_path,
                           float center_x, float center_y, float center_z,
                           float rot_angle_deg,
                           float rot_axis_x, float rot_axis_y, float rot_axis_z,
                           float scale, uint8_t flag);

/* ── VTK Export ── */

/** Export a field to a .vtk file. */
void fluidx3d_export_vtk(FluidX3D_Solver* solver, int field_id, const char* path);

/* ── Info ── */

float fluidx3d_get_nu(const FluidX3D_Solver* solver);
float fluidx3d_get_tau(const FluidX3D_Solver* solver);
float fluidx3d_get_Re_max(const FluidX3D_Solver* solver);

/** Get a default configuration matching the compile-time defines. */
FluidX3D_Config fluidx3d_default_config(void);

#ifdef __cplusplus
}
#endif

#endif /* FLUIDX3D_H */
