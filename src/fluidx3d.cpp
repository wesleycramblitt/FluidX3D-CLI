#include "fluidx3d.h"
#include "lbm.hpp"
#include <cstdlib>
#include <cstring>

struct FluidX3D_Solver {
	LBM* lbm;
	SimulationConfig config;
};

static SimulationConfig to_cpp_config(const FluidX3D_Config* c) {
	SimulationConfig cfg;
	cfg.Nx = c->nx; cfg.Ny = c->ny; cfg.Nz = c->nz;
	cfg.Dx = c->dx ? c->dx : 1u; cfg.Dy = c->dy ? c->dy : 1u; cfg.Dz = c->dz ? c->dz : 1u;
	cfg.nu = c->nu;
	cfg.fx = (c->extensions & FLUIDX3D_EXT_VOLUME_FORCE) ? c->fx : 0.0f;
	cfg.fy = (c->extensions & FLUIDX3D_EXT_VOLUME_FORCE) ? c->fy : 0.0f;
	cfg.fz = (c->extensions & FLUIDX3D_EXT_VOLUME_FORCE) ? c->fz : 0.0f;
	cfg.sigma = (c->extensions & FLUIDX3D_EXT_SURFACE) ? c->sigma : 0.0f;
	cfg.alpha = (c->extensions & FLUIDX3D_EXT_TEMPERATURE) ? (c->alpha!=0.0f ? c->alpha : 1.0f) : 0.0f;
	cfg.beta  = (c->extensions & FLUIDX3D_EXT_TEMPERATURE) ? (c->beta !=0.0f ? c->beta  : 1.0f) : 0.0f;
	cfg.particles_N = (c->extensions & FLUIDX3D_EXT_PARTICLES) ? c->particles_n : 0u;
	cfg.particles_rho = c->particles_rho;
	cfg.velocity_set = (SimulationConfig::VelocitySet)c->velocity_set;
	cfg.collision = c->collision ? SimulationConfig::CT_TRT : SimulationConfig::CT_SRT;
	cfg.precision = (SimulationConfig::PrecisionMode)c->precision;
	cfg.volume_force   = (c->extensions & FLUIDX3D_EXT_VOLUME_FORCE) != 0;
	cfg.force_field    = (c->extensions & FLUIDX3D_EXT_FORCE_FIELD) != 0;
	cfg.equilibrium_boundaries = (c->extensions & FLUIDX3D_EXT_EQUILIBRIUM_BOUNDARIES) != 0;
	cfg.moving_boundaries = (c->extensions & FLUIDX3D_EXT_MOVING_BOUNDARIES) != 0;
	cfg.surface   = (c->extensions & FLUIDX3D_EXT_SURFACE) != 0;
	cfg.temperature = (c->extensions & FLUIDX3D_EXT_TEMPERATURE) != 0;
	cfg.subgrid   = (c->extensions & FLUIDX3D_EXT_SUBGRID) != 0;
	cfg.particles = (c->extensions & FLUIDX3D_EXT_PARTICLES) != 0;
	cfg.benchmark = c->benchmark;
	cfg.si_length = c->si_length;
	cfg.si_velocity = c->si_velocity;
	cfg.si_density = c->si_density;
	if(cfg.si_length != 0.0f && cfg.si_velocity != 0.0f && cfg.si_density != 0.0f) {
		units.set_m_kg_s(1.0f, 1.0f, 1.0f, cfg.si_length, cfg.si_velocity, cfg.si_density);
	}
	if(c->device_ids[0] >= 0 && c->device_count > 0) {
		for(int i=0; i<c->device_count && i<8; i++) {
			main_arguments.push_back(to_string(c->device_ids[i]));
		}
	}
	return cfg;
}

static FluidX3D_Config to_c_config(const SimulationConfig& cfg) {
	FluidX3D_Config c = {};
	c.nx = cfg.Nx; c.ny = cfg.Ny; c.nz = cfg.Nz;
	c.dx = cfg.Dx; c.dy = cfg.Dy; c.dz = cfg.Dz;
	c.nu = cfg.nu;
	c.fx = cfg.fx; c.fy = cfg.fy; c.fz = cfg.fz;
	c.sigma = cfg.sigma;
	c.alpha = cfg.alpha; c.beta = cfg.beta;
	c.particles_n = cfg.particles_N;
	c.particles_rho = cfg.particles_rho;
	c.velocity_set = (int32_t)cfg.velocity_set;
	c.collision = (int32_t)cfg.collision;
	c.precision = (int32_t)cfg.precision;
	c.extensions = 0;
	if(cfg.volume_force) c.extensions |= FLUIDX3D_EXT_VOLUME_FORCE;
	if(cfg.force_field)  c.extensions |= FLUIDX3D_EXT_FORCE_FIELD;
	if(cfg.equilibrium_boundaries) c.extensions |= FLUIDX3D_EXT_EQUILIBRIUM_BOUNDARIES;
	if(cfg.moving_boundaries) c.extensions |= FLUIDX3D_EXT_MOVING_BOUNDARIES;
	if(cfg.surface)     c.extensions |= FLUIDX3D_EXT_SURFACE;
	if(cfg.temperature) c.extensions |= FLUIDX3D_EXT_TEMPERATURE;
	if(cfg.subgrid)     c.extensions |= FLUIDX3D_EXT_SUBGRID;
	if(cfg.particles)   c.extensions |= FLUIDX3D_EXT_PARTICLES;
	c.benchmark = cfg.benchmark;
	c.si_length = cfg.si_length;
	c.si_velocity = cfg.si_velocity;
	c.si_density = cfg.si_density;
	c.total_steps = max_ulong;
	return c;
}

FluidX3D_Config fluidx3d_default_config(void) {
	return to_c_config(SimulationConfig::from_defines());
}

FluidX3D_Solver* fluidx3d_create(const FluidX3D_Config* config) {
	if(!config) return nullptr;
	FluidX3D_Solver* s = new FluidX3D_Solver();
	s->config = to_cpp_config(config);
	try {
		s->lbm = new LBM(s->config);
		return s;
	} catch(...) {
		delete s;
		return nullptr;
	}
}

void fluidx3d_destroy(FluidX3D_Solver* solver) {
	if(!solver) return;
	delete solver->lbm;
	delete solver;
}

void fluidx3d_initialize(FluidX3D_Solver* solver) {
	if(!solver || !solver->lbm) return;
	solver->lbm->run(0u);
}

uint64_t fluidx3d_run(FluidX3D_Solver* solver, uint64_t steps, uint64_t total_steps) {
	if(!solver || !solver->lbm) return 0;
	solver->lbm->run(steps, total_steps);
	return solver->lbm->get_t();
}

void fluidx3d_set_force(FluidX3D_Solver* solver, float fx, float fy, float fz) {
	if(!solver || !solver->lbm) return;
	solver->lbm->set_f(fx, fy, fz);
}

uint64_t fluidx3d_get_step(const FluidX3D_Solver* solver) {
	if(!solver || !solver->lbm) return 0;
	return solver->lbm->get_t();
}

void fluidx3d_get_dims(const FluidX3D_Solver* solver,
                       uint32_t* nx, uint32_t* ny, uint32_t* nz) {
	if(!solver || !solver->lbm) {
		if(nx) *nx = 0; if(ny) *ny = 0; if(nz) *nz = 0;
		return;
	}
	if(nx) *nx = solver->lbm->get_Nx();
	if(ny) *ny = solver->lbm->get_Ny();
	if(nz) *nz = solver->lbm->get_Nz();
}

uint64_t fluidx3d_field_size(const FluidX3D_Solver* solver, int field_id) {
	if(!solver || !solver->lbm) return 0;
	ulong N = solver->lbm->get_N();
	switch(field_id) {
		case FLUIDX3D_FIELD_RHO:       return N;
		case FLUIDX3D_FIELD_U:         return 3*N;
		case FLUIDX3D_FIELD_FLAGS:     return N;
		case FLUIDX3D_FIELD_FORCE:     return 3*N;
		case FLUIDX3D_FIELD_PHI:       return N;
		case FLUIDX3D_FIELD_TEMP:      return N;
		case FLUIDX3D_FIELD_PARTICLES: return 3*solver->config.particles_N;
		default: return 0;
	}
}

int fluidx3d_field_components(int field_id) {
	switch(field_id) {
		case FLUIDX3D_FIELD_RHO:  return 1;
		case FLUIDX3D_FIELD_U:    return 3;
		case FLUIDX3D_FIELD_FLAGS: return 1;
		case FLUIDX3D_FIELD_FORCE: return 3;
		case FLUIDX3D_FIELD_PHI:  return 1;
		case FLUIDX3D_FIELD_TEMP: return 1;
		case FLUIDX3D_FIELD_PARTICLES: return 3;
		default: return 0;
	}
}

uint64_t fluidx3d_read_field(FluidX3D_Solver* solver, int field_id,
                             void* host_buffer, uint64_t buffer_size) {
	if(!solver || !solver->lbm || !host_buffer) return 0;
	uint64_t needed = fluidx3d_field_size(solver, field_id);
	if(buffer_size < needed) return 0;
	ulong N = solver->lbm->get_N();

	switch(field_id) {
		case FLUIDX3D_FIELD_RHO: {
			solver->lbm->rho.read_from_device();
			float* dst = (float*)host_buffer;
			for(ulong i=0; i<N; i++) dst[i] = solver->lbm->rho[i];
			return N;
		}
		case FLUIDX3D_FIELD_U: {
			solver->lbm->u.read_from_device();
			float* dst = (float*)host_buffer;
			for(ulong i=0; i<N; i++) {
				dst[i]   = solver->lbm->u.x[i];
				dst[i+N] = solver->lbm->u.y[i];
				dst[i+2*N] = solver->lbm->u.z[i];
			}
			return 3*N;
		}
		case FLUIDX3D_FIELD_FLAGS: {
			solver->lbm->flags.read_from_device();
			memcpy(host_buffer, &solver->lbm->flags[0], N*sizeof(uchar));
			return N;
		}
#ifdef FORCE_FIELD
		case FLUIDX3D_FIELD_FORCE: {
			solver->lbm->F.read_from_device();
			float* dst = (float*)host_buffer;
			for(ulong i=0; i<N; i++) {
				dst[i]   = solver->lbm->F.x[i];
				dst[i+N] = solver->lbm->F.y[i];
				dst[i+2*N] = solver->lbm->F.z[i];
			}
			return 3*N;
		}
#else
		case FLUIDX3D_FIELD_FORCE: return 0;
#endif
#ifdef SURFACE
		case FLUIDX3D_FIELD_PHI: {
			solver->lbm->phi.read_from_device();
			float* dst = (float*)host_buffer;
			for(ulong i=0; i<N; i++) dst[i] = solver->lbm->phi[i];
			return N;
		}
#else
		case FLUIDX3D_FIELD_PHI: return 0;
#endif
#ifdef TEMPERATURE
		case FLUIDX3D_FIELD_TEMP: {
			solver->lbm->T.read_from_device();
			float* dst = (float*)host_buffer;
			for(ulong i=0; i<N; i++) dst[i] = solver->lbm->T[i];
			return N;
		}
#else
		case FLUIDX3D_FIELD_TEMP: return 0;
#endif
#ifdef PARTICLES
		case FLUIDX3D_FIELD_PARTICLES: {
			ulong pN = solver->config.particles_N;
			if(pN == 0) return 0;
			float* dst = (float*)host_buffer;
			for(ulong i=0; i<pN; i++) {
				dst[i]     = solver->lbm->particles->x[i];
				dst[i+pN]   = solver->lbm->particles->y[i];
				dst[i+2*pN] = solver->lbm->particles->z[i];
			}
			return 3*pN;
		}
#else
		case FLUIDX3D_FIELD_PARTICLES: return 0;
#endif
		default: return 0;
	}
}

void fluidx3d_write_field(FluidX3D_Solver* solver, int field_id,
                          const void* host_buffer, uint64_t buffer_size) {
	if(!solver || !solver->lbm || !host_buffer) return;
	ulong N = solver->lbm->get_N();

	switch(field_id) {
		case FLUIDX3D_FIELD_RHO: {
			const float* src = (const float*)host_buffer;
			for(ulong i=0; i<N && i<buffer_size; i++) solver->lbm->rho[i] = src[i];
			solver->lbm->rho.write_to_device();
			break;
		}
		case FLUIDX3D_FIELD_U: {
			const float* src = (const float*)host_buffer;
			for(ulong i=0; i<N && i*3<buffer_size; i++) {
				solver->lbm->u.x[i] = src[i];
				solver->lbm->u.y[i] = src[i+N];
				solver->lbm->u.z[i] = src[i+2*N];
			}
			solver->lbm->u.write_to_device();
			break;
		}
		case FLUIDX3D_FIELD_FLAGS: {
			memcpy(&solver->lbm->flags[0], host_buffer, std::min((ulong)buffer_size, N*sizeof(uchar)));
			solver->lbm->flags.write_to_device();
			break;
		}
#ifdef FORCE_FIELD
		case FLUIDX3D_FIELD_FORCE: {
			const float* src = (const float*)host_buffer;
			for(ulong i=0; i<N && i*3<buffer_size; i++) {
				solver->lbm->F.x[i] = src[i];
				solver->lbm->F.y[i] = src[i+N];
				solver->lbm->F.z[i] = src[i+2*N];
			}
			solver->lbm->F.write_to_device();
			break;
		}
#endif
#ifdef SURFACE
		case FLUIDX3D_FIELD_PHI: {
			const float* src = (const float*)host_buffer;
			for(ulong i=0; i<N && i<buffer_size; i++) solver->lbm->phi[i] = src[i];
			solver->lbm->phi.write_to_device();
			break;
		}
#endif
#ifdef TEMPERATURE
		case FLUIDX3D_FIELD_TEMP: {
			const float* src = (const float*)host_buffer;
			for(ulong i=0; i<N && i<buffer_size; i++) solver->lbm->T[i] = src[i];
			solver->lbm->T.write_to_device();
			break;
		}
#endif
	}
}

float* fluidx3d_read_field_float(FluidX3D_Solver* solver, int field_id,
                                 uint64_t* out_count) {
	uint64_t count = fluidx3d_field_size(solver, field_id);
	if(count == 0) { if(out_count) *out_count = 0; return nullptr; }
	float* buf = (float*)malloc(count * sizeof(float));
	if(!buf) { if(out_count) *out_count = 0; return nullptr; }
	uint64_t read = fluidx3d_read_field(solver, field_id, buf, count);
	if(read != count) { free(buf); if(out_count) *out_count = 0; return nullptr; }
	if(out_count) *out_count = count;
	return buf;
}

uint8_t* fluidx3d_read_field_uchar(FluidX3D_Solver* solver, int field_id,
                                   uint64_t* out_count) {
	if(field_id != FLUIDX3D_FIELD_FLAGS) { if(out_count) *out_count = 0; return nullptr; }
	uint64_t count = fluidx3d_field_size(solver, field_id);
	if(count == 0) { if(out_count) *out_count = 0; return nullptr; }
	uint8_t* buf = (uint8_t*)malloc(count);
	if(!buf) { if(out_count) *out_count = 0; return nullptr; }
	uint64_t read = fluidx3d_read_field(solver, field_id, buf, count);
	if(read != count) { free(buf); if(out_count) *out_count = 0; return nullptr; }
	if(out_count) *out_count = count;
	return buf;
}

void fluidx3d_voxelize_stl(FluidX3D_Solver* solver,
                           const char* stl_path,
                           float center_x, float center_y, float center_z,
                           float rot_angle_deg,
                           float rot_axis_x, float rot_axis_y, float rot_axis_z,
                           float scale, uint8_t flag) {
	if(!solver || !solver->lbm || !stl_path) return;
	float3 center = float3(center_x, center_y, center_z);
	float3x3 rotation = float3x3(float3(rot_axis_x, rot_axis_y, rot_axis_z), radians(rot_angle_deg));
	solver->lbm->voxelize_stl(string(stl_path), center, rotation, scale, flag);
}

void fluidx3d_export_vtk(FluidX3D_Solver* solver, int field_id, const char* path) {
	if(!solver || !solver->lbm || !path) return;
	switch(field_id) {
		case FLUIDX3D_FIELD_RHO:   solver->lbm->rho.write_device_to_vtk(path); break;
		case FLUIDX3D_FIELD_U:     solver->lbm->u.write_device_to_vtk(path); break;
		case FLUIDX3D_FIELD_FLAGS: solver->lbm->flags.write_device_to_vtk(path); break;
#ifdef FORCE_FIELD
		case FLUIDX3D_FIELD_FORCE: solver->lbm->F.write_device_to_vtk(path); break;
#endif
#ifdef SURFACE
		case FLUIDX3D_FIELD_PHI:   solver->lbm->phi.write_device_to_vtk(path); break;
#endif
#ifdef TEMPERATURE
		case FLUIDX3D_FIELD_TEMP:  solver->lbm->T.write_device_to_vtk(path); break;
#endif
	}
}

float fluidx3d_get_nu(const FluidX3D_Solver* solver)  { return solver&&solver->lbm ? solver->lbm->get_nu() : 0.0f; }
float fluidx3d_get_tau(const FluidX3D_Solver* solver) { return solver&&solver->lbm ? solver->lbm->get_tau() : 0.0f; }
float fluidx3d_get_Re_max(const FluidX3D_Solver* solver) { return solver&&solver->lbm ? solver->lbm->get_Re_max() : 0.0f; }

int fluidx3d_get_device_count(const FluidX3D_Solver* solver) {
	if(!solver || !solver->lbm) return 0;
	return (int)solver->lbm->get_D();
}

intptr_t fluidx3d_get_cl_mem_handle(const FluidX3D_Solver* solver, int field_id,
                                     int device_index, int component,
                                     uint64_t* out_size_bytes) {
	if(!solver || !solver->lbm) return 0;
	if(device_index < 0 || (uint)device_index >= (int)solver->lbm->get_D()) return 0;

	LBM_Domain* dom = solver->lbm->lbm_domain[device_index];

	// For velocity (U), each component is a separate buffer (x, y, z)
	if(field_id == FLUIDX3D_FIELD_U && component >= 0 && component <= 2) {
		// u is Memory<float> with dimensions=3, stored interleaved per-domain.
		// The device buffer contains all 3 components. For component access,
		// we return the whole buffer and tell the caller the per-component offset.
		Memory<float>* mem = &dom->u;
		ulong N = dom->get_N();
		if(out_size_bytes) *out_size_bytes = N * sizeof(float);
		// The caller needs to offset by component * N * sizeof(float) when reading
		// For raw cl_mem, return the base buffer handle
		return (intptr_t)mem->get_cl_mem_handle();
	}
#ifdef FORCE_FIELD
	if(field_id == FLUIDX3D_FIELD_FORCE && component >= 0 && component <= 2) {
		Memory<float>* mem = &dom->F;
		ulong N = dom->get_N();
		if(out_size_bytes) *out_size_bytes = N * sizeof(float);
		return (intptr_t)mem->get_cl_mem_handle();
	}
#endif

	// Scalar fields
	if(field_id == FLUIDX3D_FIELD_RHO) {
		ulong N = dom->get_N();
		if(out_size_bytes) *out_size_bytes = N * sizeof(float);
		return (intptr_t)dom->rho.get_cl_mem_handle();
	}
	if(field_id == FLUIDX3D_FIELD_FLAGS) {
		ulong N = dom->get_N();
		if(out_size_bytes) *out_size_bytes = N * sizeof(uchar);
		return (intptr_t)dom->flags.get_cl_mem_handle();
	}
#ifdef SURFACE
	if(field_id == FLUIDX3D_FIELD_PHI) {
		ulong N = dom->get_N();
		if(out_size_bytes) *out_size_bytes = N * sizeof(float);
		return (intptr_t)dom->phi.get_cl_mem_handle();
	}
#endif
#ifdef TEMPERATURE
	if(field_id == FLUIDX3D_FIELD_TEMP) {
		ulong N = dom->get_N();
		if(out_size_bytes) *out_size_bytes = N * sizeof(float);
		return (intptr_t)dom->T.get_cl_mem_handle();
	}
#endif
	return 0;
}

intptr_t fluidx3d_get_cl_queue_handle(const FluidX3D_Solver* solver, int device_index) {
	if(!solver || !solver->lbm) return 0;
	if(device_index < 0 || (uint)device_index >= solver->lbm->get_D()) return 0;
	// cl::CommandQueue stores cl_command_queue as first member
	const cl::CommandQueue& q = solver->lbm->lbm_domain[device_index]->get_device().get_cl_queue();
	return (intptr_t)*(cl_command_queue*)(void*)&q;
}
