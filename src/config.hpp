#pragma once

#include "defines.hpp"
#include "utilities.hpp"

// ── Runtime simulation configuration ──────────────────────────────────────────

struct SimulationConfig {
	// Grid
	uint Nx=256u, Ny=256u, Nz=256u;
	uint Dx=1u,  Dy=1u,  Dz=1u;

	// Physics
	float nu = 1.0f/6.0f;
	float fx=0.0f, fy=0.0f, fz=0.0f;
	float sigma=0.0f;
	float alpha=1.0f, beta=1.0f;
	uint  particles_N=0u;
	float particles_rho=1.0f;

	// LBM variant
	enum VelocitySet { VS_D2Q9=9, VS_D3Q15=15, VS_D3Q19=19, VS_D3Q27=27 };
	VelocitySet velocity_set = VS_D3Q19;

	enum CollisionType { CT_SRT=0, CT_TRT=1 };
	CollisionType collision = CT_SRT;

	enum PrecisionMode { PM_FP32=0, PM_FP16S=1, PM_FP16C=2 };
	PrecisionMode precision = PM_FP32;

	// Unit conversion (0 = use LBM units)
	float si_length=0.0f, si_velocity=0.0f, si_density=0.0f;

	// Extensions
	bool volume_force         = false;
	bool force_field          = false;
	bool equilibrium_boundaries = false;
	bool moving_boundaries    = false;
	bool surface              = false;
	bool temperature          = false;
	bool subgrid              = false;
	bool particles            = false;
	bool benchmark            = false;

	// Derived helpers
	uint dimensions() const { return velocity_set==VS_D2Q9 ? 2u : 3u; }
	uint transfers() const {
		switch(velocity_set) {
			case VS_D2Q9:  return 3u;
			case VS_D3Q15: return 5u;
			case VS_D3Q19: return 5u;
			case VS_D3Q27: return 9u;
		}
		return 5u;
	}
	bool update_fields() const { return surface || particles; }

	// Apply defaults from classic compile-time defines (if nothing is explicitly set)
	static SimulationConfig from_defines();
};

inline SimulationConfig SimulationConfig::from_defines() {
	SimulationConfig cfg;
#if defined(D2Q9)
	cfg.velocity_set = VS_D2Q9;
#elif defined(D3Q15)
	cfg.velocity_set = VS_D3Q15;
#elif defined(D3Q27)
	cfg.velocity_set = VS_D3Q27;
#else
	cfg.velocity_set = VS_D3Q19; // default
#endif
#if defined(TRT)
	cfg.collision = CT_TRT;
#else
	cfg.collision = CT_SRT; // default
#endif
#if defined(FP16C)
	cfg.precision = PM_FP16C;
#elif defined(FP16S)
	cfg.precision = PM_FP16S;
#else
	cfg.precision = PM_FP32; // default
#endif
#ifdef VOLUME_FORCE
	cfg.volume_force = true;
#endif
#ifdef FORCE_FIELD
	cfg.force_field = true;
#endif
#ifdef EQUILIBRIUM_BOUNDARIES
	cfg.equilibrium_boundaries = true;
#endif
#ifdef MOVING_BOUNDARIES
	cfg.moving_boundaries = true;
#endif
#ifdef SURFACE
	cfg.surface = true;
#endif
#ifdef TEMPERATURE
	cfg.temperature = true;
#endif
#ifdef SUBGRID
	cfg.subgrid = true;
#endif
#ifdef PARTICLES
	cfg.particles = true;
#endif
#ifdef BENCHMARK
	cfg.benchmark = true;
#endif
	return cfg;
}
