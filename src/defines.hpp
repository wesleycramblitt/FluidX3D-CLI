#pragma once



//#define D2Q9 // choose D2Q9 velocity set for 2D; allocates 53 (FP32) or 35 (FP16) Bytes/cell
//#define D3Q15 // choose D3Q15 velocity set for 3D; allocates 77 (FP32) or 47 (FP16) Bytes/cell
#define D3Q19 // choose D3Q19 velocity set for 3D; allocates 93 (FP32) or 55 (FP16) Bytes/cell; (default)
//#define D3Q27 // choose D3Q27 velocity set for 3D; allocates 125 (FP32) or 71 (FP16) Bytes/cell

#define SRT // choose single-relaxation-time LBM collision operator; (default)
//#define TRT // choose two-relaxation-time LBM collision operator

#define FP16S // optional for 2x speedup and 2x VRAM footprint reduction: compress LBM DDFs to range-shifted IEEE-754 FP16; number conversion is done in hardware; all arithmetic is still done in FP32
//#define FP16C // optional for 2x speedup and 2x VRAM footprint reduction: compress LBM DDFs to more accurate custom FP16C format; number conversion is emulated in software; all arithmetic is still done in FP32

//#define BENCHMARK // disable all extensions and setups and run benchmark setup instead

//#define VOLUME_FORCE // enables global force per volume in one direction (equivalent to a pressure gradient); specified in the LBM class constructor; the force can be changed on-the-fly between time steps at no performance cost
//#define FORCE_FIELD // enables computing the forces on solid boundaries with lbm.update_force_field(); and enables setting the force for each lattice point independently (enable VOLUME_FORCE too); allocates an extra 12 Bytes/cell
//#define EQUILIBRIUM_BOUNDARIES // enables fixing the velocity/density by marking cells with TYPE_E; can be used for inflow/outflow; does not reflect shock waves
//#define MOVING_BOUNDARIES // enables moving solids: set solid cells to TYPE_S and set their velocity u unequal to zero
//#define SURFACE // enables free surface LBM: mark fluid cells with TYPE_F; at initialization the TYPE_I interface and TYPE_G gas domains will automatically be completed; allocates an extra 12 Bytes/cell
//#define TEMPERATURE // enables temperature extension; set fixed-temperature cells with TYPE_T (similar to EQUILIBRIUM_BOUNDARIES); allocates an extra 32 (FP32) or 18 (FP16) Bytes/cell
//#define SUBGRID // enables Smagorinsky-Lilly subgrid turbulence LES model to keep simulations with very large Reynolds number stable
//#define PARTICLES // enables particles with immersed-boundary method (for 2-way coupling also activate VOLUME_FORCE and FORCE_FIELD; only supported in single-GPU)

#define TYPE_S 0b00000001 // (stationary or moving) solid boundary
#define TYPE_E 0b00000010 // equilibrium boundary (inflow/outflow)
#define TYPE_T 0b00000100 // temperature boundary
#define TYPE_F 0b00001000 // fluid
#define TYPE_I 0b00010000 // interface
#define TYPE_G 0b00100000 // gas
#define TYPE_X 0b01000000 // reserved type X
#define TYPE_Y 0b10000000 // reserved type Y

#if defined(FP16S) || defined(FP16C)
#define fpxx ushort
#else // FP32
#define fpxx float
#endif // FP32

#ifdef BENCHMARK
#undef UPDATE_FIELDS
#undef VOLUME_FORCE
#undef FORCE_FIELD
#undef MOVING_BOUNDARIES
#undef EQUILIBRIUM_BOUNDARIES
#undef SURFACE
#undef TEMPERATURE
#undef SUBGRID
#undef PARTICLES
#endif // BENCHMARK

#ifdef SURFACE // (rho, u) need to be updated exactly every LBM step
#define UPDATE_FIELDS // update (rho, u, T) in every LBM step
#endif // SURFACE

#ifdef TEMPERATURE
#define VOLUME_FORCE
#endif // TEMPERATURE

#ifdef PARTICLES // (rho, u) need to be updated exactly every LBM step
#define UPDATE_FIELDS // update (rho, u, T) in every LBM step
#endif // PARTICLES
