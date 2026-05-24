#include "lbm.hpp"



Units units; // for unit conversion

// Compile-time velocity set constants (used by backward-compat constructors)
#if defined(D2Q9)
static const uint velocity_set = 9u;
static const uint dimensions = 2u;
static const uint transfers = 3u;
#elif defined(D3Q15)
static const uint velocity_set = 15u;
static const uint dimensions = 3u;
static const uint transfers = 5u;
#elif defined(D3Q19)
static const uint velocity_set = 19u;
static const uint dimensions = 3u;
static const uint transfers = 5u;
#elif defined(D3Q27)
static const uint velocity_set = 27u;
static const uint dimensions = 3u;
static const uint transfers = 9u;
#endif

uint bytes_per_cell_host(const SimulationConfig& cfg) { // returns the number of Bytes per cell allocated in host memory
	uint bytes_per_cell = 17u; // rho, u, flags
	if(cfg.force_field) bytes_per_cell += 12u;
	if(cfg.surface)     bytes_per_cell += 4u;
	if(cfg.temperature) bytes_per_cell += 4u;
	return bytes_per_cell;
}
uint bytes_per_cell_device(const SimulationConfig& cfg) { // returns the number of Bytes per cell allocated in device memory
	uint bytes_per_cell = cfg.velocity_set*sizeof(fpxx)+17u; // fi, rho, u, flags
	if(cfg.force_field) bytes_per_cell += 12u;
	if(cfg.surface)     bytes_per_cell += 12u;
	if(cfg.temperature) bytes_per_cell += 7u*sizeof(fpxx)+4u;
	return bytes_per_cell;
}
uint bandwidth_bytes_per_cell_device(const SimulationConfig& cfg) { // returns the bandwidth in Bytes per cell per time step from/to device memory
	uint bandwidth_bytes_per_cell = cfg.velocity_set*2u*sizeof(fpxx)+1u;
	if(cfg.update_fields()) {
		bandwidth_bytes_per_cell += 16u;
		if(cfg.temperature) bandwidth_bytes_per_cell += 4u;
	}
	if(cfg.force_field) bandwidth_bytes_per_cell += 12u;
	if(cfg.moving_boundaries || cfg.surface || cfg.temperature)
		bandwidth_bytes_per_cell += (cfg.velocity_set-1u)*1u;
	if(cfg.surface)
		bandwidth_bytes_per_cell += (1u+(2u*cfg.velocity_set-1u)*sizeof(fpxx)+8u+(cfg.velocity_set-1u)*4u) + 1u + 1u + (4u+cfg.velocity_set+4u+4u+4u);
	if(cfg.temperature)
		bandwidth_bytes_per_cell += 7u*2u*sizeof(fpxx);
	return bandwidth_bytes_per_cell;
}
uint3 resolution(const float3 box_aspect_ratio, const uint memory, const SimulationConfig& cfg) { // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	if(cfg.dimensions()==2u) {
		float memory_required = (box_aspect_ratio.x*box_aspect_ratio.y)*(float)bytes_per_cell_device(cfg)/1048576.0f; // in MB
		float scaling = sqrt((float)memory/memory_required);
		return uint3(to_uint(scaling*box_aspect_ratio.x), to_uint(scaling*box_aspect_ratio.y), 1u);
	} else {
		float memory_required = (box_aspect_ratio.x*box_aspect_ratio.y*box_aspect_ratio.z)*(float)bytes_per_cell_device(cfg)/1048576.0f; // in MB
		float scaling = cbrt((float)memory/memory_required);
		return uint3(to_uint(scaling*box_aspect_ratio.x), to_uint(scaling*box_aspect_ratio.y), to_uint(scaling*box_aspect_ratio.z));
	}
}

string default_filename(const string& path, const string& name, const string& extension, const ulong t) { // generate a default filename with timestamp
	string time = "00000000"+to_string(t);
	time = substring(time, length(time)-9u, 9u);
	return (path=="" ? get_exe_path()+"export/" : path)+create_file_extension((name=="" ? "file" : name)+"-"+time, extension);
}
string default_filename(const string& name, const string& extension, const ulong t) { // generate a default filename with timestamp at exe_path/export/
	return default_filename("", name, extension, t);
}



LBM_Domain::LBM_Domain(const Device_Info& device_info, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const int Ox, const int Oy, const int Oz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // constructor with manual device selection and domain offset
	this->Nx = Nx; this->Ny = Ny; this->Nz = Nz;
	this->Dx = Dx; this->Dy = Dy; this->Dz = Dz;
	this->Ox = Ox; this->Oy = Oy; this->Oz = Oz;
	this->nu = nu;
	this->fx = fx; this->fy = fy; this->fz = fz;
	this->sigma = sigma;
	this->alpha = alpha; this->beta = beta;
	this->particles_N = particles_N;
	this->particles_rho = particles_rho;
	// Set runtime extension flags from compile-time defines (backward compat)
	this->velocity_set = ::velocity_set; this->dims = (uint)dimensions; this->trans = (uint)transfers;
	this->fp_precision = 0;
#ifdef FP16S
	this->fp_precision = 1;
#endif
#ifdef FP16C
	this->fp_precision = 2;
#endif
#ifdef TRT
	this->use_srt = false;
#endif
#ifdef VOLUME_FORCE
	this->has_volume_force = true;
#endif
#ifdef FORCE_FIELD
	this->has_force_field = true;
#endif
#ifdef EQUILIBRIUM_BOUNDARIES
	this->has_equilibrium_boundaries = true;
#endif
#ifdef MOVING_BOUNDARIES
	this->has_moving_boundaries = true;
#endif
#ifdef SURFACE
	this->has_surface = true;
#endif
#ifdef TEMPERATURE
	this->has_temperature = true;
#endif
#ifdef SUBGRID
	this->has_subgrid = true;
#endif
#ifdef PARTICLES
	this->has_particles = true;
#endif
	this->has_update_fields = this->has_surface || this->has_particles;
#ifdef UPDATE_FIELDS
	this->has_update_fields = true;
#endif
	string opencl_c_code;
	opencl_c_code = device_defines(device_info)+get_opencl_c_code();
	this->device = Device(device_info, opencl_c_code);
	print_info("Allocating memory. This may take a few seconds.");
	allocate(device); // lbm first
}
LBM_Domain::LBM_Domain(const Device_Info& device_info, const SimulationConfig& config, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const int Ox, const int Oy, const int Oz) { // constructor with runtime config
	this->Nx = Nx; this->Ny = Ny; this->Nz = Nz;
	this->Dx = Dx; this->Dy = Dy; this->Dz = Dz;
	this->Ox = Ox; this->Oy = Oy; this->Oz = Oz;
	this->nu = config.nu;
	this->fx = config.fx; this->fy = config.fy; this->fz = config.fz;
	this->sigma = config.sigma;
	this->alpha = config.alpha; this->beta = config.beta;
	this->particles_N = config.particles_N;
	this->particles_rho = config.particles_rho;
	this->velocity_set = (uint)config.velocity_set;
	this->dims = config.dimensions();
	this->trans = config.transfers();
	this->use_srt = (config.collision == SimulationConfig::CT_SRT);
	this->fp_precision = (int)config.precision;
	this->has_volume_force = config.volume_force;
	this->has_force_field = config.force_field;
	this->has_equilibrium_boundaries = config.equilibrium_boundaries;
	this->has_moving_boundaries = config.moving_boundaries;
	this->has_surface = config.surface;
	this->has_temperature = config.temperature;
	this->has_subgrid = config.subgrid;
	this->has_particles = config.particles;
	this->has_update_fields = config.update_fields();
	string opencl_c_code;
	opencl_c_code = device_defines(device_info)+get_opencl_c_code();
	this->device = Device(device_info, opencl_c_code);
	print_info("Allocating memory. This may take a few seconds.");
	allocate(device); // lbm first
}

void LBM_Domain::allocate(Device& device) {
	const ulong N = get_N();
	fi = Memory<fpxx>(device, N, velocity_set, false);
	rho = Memory<float>(device, N, 1u, true, true, 1.0f);
	u = Memory<float>(device, N, 3u);
	flags = Memory<uchar>(device, N);
	kernel_initialize = Kernel(device, N, "initialize", fi, rho, u, flags);
	kernel_stream_collide = Kernel(device, N, "stream_collide", fi, rho, u, flags, t, fx, fy, fz);
	kernel_update_fields = Kernel(device, N, "update_fields", fi, rho, u, flags, t, fx, fy, fz);

	if(has_force_field) {
		F = Memory<float>(device, N, 3u);
		object_sum = Memory<float>(device, 1u, 4u);
		kernel_stream_collide.add_parameters(F);
		kernel_update_fields.add_parameters(F);
		kernel_update_force_field = Kernel(device, N, "update_force_field", fi, flags, t, F);
		kernel_reset_force_field = Kernel(device, N, "reset_force_field", F);
		kernel_object_center_of_mass = Kernel(device, N, "object_center_of_mass", flags, (uchar)0u, object_sum);
		kernel_object_force = Kernel(device, N, "object_force", F, flags, (uchar)0u, object_sum);
		kernel_object_torque = Kernel(device, N, "object_torque", F, flags, (uchar)0u, 0.0f, 0.0f, 0.0f, object_sum);
	}

	if(has_moving_boundaries) {
		kernel_update_moving_boundaries = Kernel(device, N, "update_moving_boundaries", u, flags);
	}

	if(has_surface) {
		phi = Memory<float>(device, N);
		mass = Memory<float>(device, N, 1u, false);
		massex = Memory<float>(device, N, 1u, false);
		kernel_initialize.add_parameters(mass, massex, phi);
		kernel_stream_collide.add_parameters(mass);
		kernel_surface_0 = Kernel(device, N, "surface_0", fi, rho, u, flags, mass, massex, phi, t, fx, fy, fz);
		kernel_surface_1 = Kernel(device, N, "surface_1", flags);
		kernel_surface_2 = Kernel(device, N, "surface_2", fi, rho, u, flags, t);
		kernel_surface_3 = Kernel(device, N, "surface_3", rho, flags, mass, massex, phi);
	}

	if(has_temperature) {
		gi = Memory<fpxx>(device, N, 7u, false);
		T = Memory<float>(device, N, 1u, true, true, 1.0f);
		kernel_initialize.add_parameters(gi, T);
		kernel_stream_collide.add_parameters(gi, T);
		kernel_update_fields.add_parameters(gi, T);
	}

	if(has_particles) {
		particles = Memory<float>(device, (ulong)particles_N, 3u);
		kernel_integrate_particles = Kernel(device, (ulong)particles_N, "integrate_particles", particles, u, flags, 1.0f);
		if(has_force_field) {
			kernel_integrate_particles.add_parameters(F, fx, fy, fz);
		}
	}

	if(get_D()>1u) allocate_transfer(device);
}

void LBM_Domain::enqueue_initialize() { // call kernel_initialize
	kernel_initialize.enqueue_run();
}
void LBM_Domain::enqueue_stream_collide() { // call kernel_stream_collide to perform one LBM time step
	kernel_stream_collide.set_parameters(4u, t, fx, fy, fz).enqueue_run();
}
void LBM_Domain::enqueue_update_fields() { // update fields (rho, u, T) manually
#ifndef UPDATE_FIELDS
	if(t!=t_last_update_fields) { // only run kernel_update_fields if the time step has changed since last update
		kernel_update_fields.set_parameters(4u, t, fx, fy, fz).enqueue_run();
		t_last_update_fields = t;
	}
#endif // UPDATE_FIELDS
}
#ifdef SURFACE
void LBM_Domain::enqueue_surface_0() {
	kernel_surface_0.set_parameters(7u, t, fx, fy, fz).enqueue_run();
}
void LBM_Domain::enqueue_surface_1() {
	kernel_surface_1.enqueue_run();
}
void LBM_Domain::enqueue_surface_2() {
	kernel_surface_2.set_parameters(4u, t).enqueue_run();
}
void LBM_Domain::enqueue_surface_3() {
	kernel_surface_3.enqueue_run();
}
#endif // SURFACE
#ifdef FORCE_FIELD
void LBM_Domain::enqueue_update_force_field() { // calculate forces from fluid on TYPE_S cells
	if(t!=t_last_force_field) { // only run kernel_update_force_field if the time step has changed since last update
		kernel_update_force_field.set_parameters(2u, t).enqueue_run();
		t_last_force_field = t;
	}
}
void LBM_Domain::enqueue_object_center_of_mass(const uchar flag_marker) { // calculate center of mass of all cells flagged with flag_marker
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_center_of_mass.set_parameters(1u, flag_marker).enqueue_run();
	object_sum.enqueue_read_from_device();
}
void LBM_Domain::enqueue_object_force(const uchar flag_marker) { // add up force for all cells flagged with flag_marker
	enqueue_update_force_field(); // update force field if it is not yet up-to-date
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_force.set_parameters(2u, flag_marker).enqueue_run();
	object_sum.enqueue_read_from_device();
}
void LBM_Domain::enqueue_object_torque(const float3& rotation_center, const uchar flag_marker) { // add up torque around specified rotation_center for all cells flagged with flag_marker
	enqueue_update_force_field(); // update force field if it is not yet up-to-date
	object_sum.x[0] = 0.0f; // reset object_sum
	object_sum.y[0] = 0.0f;
	object_sum.z[0] = 0.0f;
	object_sum.enqueue_write_to_device();
	kernel_object_torque.set_parameters(2u, flag_marker, rotation_center.x, rotation_center.y, rotation_center.z).enqueue_run();
	object_sum.enqueue_read_from_device();
}
#endif // FORCE_FIELD
#ifdef MOVING_BOUNDARIES
void LBM_Domain::enqueue_update_moving_boundaries() { // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	kernel_update_moving_boundaries.enqueue_run();
}
#endif // MOVING_BOUNDARIES
#ifdef PARTICLES
void LBM_Domain::enqueue_integrate_particles(const uint time_step_multiplicator) { // intgegrate particles forward in time and couple particles to fluid
#ifdef FORCE_FIELD
	if(particles_rho!=1.0f) kernel_reset_force_field.enqueue_run(); // only reset force field if particles have buoyancy and apply forces on fluid
	kernel_integrate_particles.set_parameters(5u, fx, fy, fz);
#endif // FORCE_FIELD
	kernel_integrate_particles.set_parameters(3u, (float)time_step_multiplicator).enqueue_run();
}
#endif // PARTICLES

void LBM_Domain::increment_time_step(const ulong steps) {
	t += steps; // increment time step
#ifdef UPDATE_FIELDS
	t_last_update_fields = t;
#endif // UPDATE_FIELDS
}
void LBM_Domain::reset_time_step() {
	t = 0ull; // increment time step
#ifdef UPDATE_FIELDS
	t_last_update_fields = t;
#endif // UPDATE_FIELDS
}
void LBM_Domain::finish_queue() {
	device.finish_queue();
}

void LBM_Domain::voxelize_mesh_on_device(const Mesh* mesh, const uchar flag, const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity) { // voxelize triangle mesh
	Memory<float3> p0(device, mesh->triangle_number, 1u, mesh->p0);
	Memory<float3> p1(device, mesh->triangle_number, 1u, mesh->p1);
	Memory<float3> p2(device, mesh->triangle_number, 1u, mesh->p2);
	Memory<float> bounding_box_and_velocity(device, 16u);
	const float x0=mesh->pmin.x-2.0f, y0=mesh->pmin.y-2.0f, z0=mesh->pmin.z-2.0f, x1=mesh->pmax.x+2.0f, y1=mesh->pmax.y+2.0f, z1=mesh->pmax.z+2.0f; // use bounding box of mesh to speed up voxelization; add tolerance of 2 cells for re-voxelization of moving objects
	bounding_box_and_velocity[ 0] = as_float(mesh->triangle_number);
	bounding_box_and_velocity[ 1] = x0;
	bounding_box_and_velocity[ 2] = y0;
	bounding_box_and_velocity[ 3] = z0;
	bounding_box_and_velocity[ 4] = x1;
	bounding_box_and_velocity[ 5] = y1;
	bounding_box_and_velocity[ 6] = z1;
	bounding_box_and_velocity[ 7] = rotation_center.x;
	bounding_box_and_velocity[ 8] = rotation_center.y;
	bounding_box_and_velocity[ 9] = rotation_center.z;
	bounding_box_and_velocity[10] = linear_velocity.x;
	bounding_box_and_velocity[11] = linear_velocity.y;
	bounding_box_and_velocity[12] = linear_velocity.z;
	bounding_box_and_velocity[13] = rotational_velocity.x;
	bounding_box_and_velocity[14] = rotational_velocity.y;
	bounding_box_and_velocity[15] = rotational_velocity.z;
	uint direction = 0u;
	if(length(rotational_velocity)==0.0f) { // choose direction of minimum bounding-box cross-section area
		float v[3] = { (y1-y0)*(z1-z0), (z1-z0)*(x1-x0), (x1-x0)*(y1-y0) };
		float vmin = v[0];
		for(uint i=1u; i<3u; i++) {
			if(v[i]<vmin) {
				vmin = v[i];
				direction = i;
			}
		}
	} else { // choose direction closest to rotation axis
		float v[3] = { fabsf(rotational_velocity.x), fabsf(rotational_velocity.y), fabsf(rotational_velocity.z) };
		float vmax = v[0];
		for(uint i=1u; i<3u; i++) {
			if(v[i]>vmax) {
				vmax = v[i];
				direction = i; // find direction of minimum bounding-box cross-section area
			}
		}
	}
	const ulong A[3] = { (ulong)Ny*(ulong)Nz, (ulong)Nz*(ulong)Nx, (ulong)Nx*(ulong)Ny };
	Kernel kernel_voxelize_mesh(device, A[direction], "voxelize_mesh", direction, fi, u, flags, t+1ull, flag, p0, p1, p2, bounding_box_and_velocity);
#ifdef SURFACE
	kernel_voxelize_mesh.add_parameters(mass, massex);
#endif // SURFACE
	p0.write_to_device();
	p1.write_to_device();
	p2.write_to_device();
	bounding_box_and_velocity.write_to_device();
	kernel_voxelize_mesh.run();
}
void LBM_Domain::enqueue_unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag) { // remove voxelized triangle mesh from LBM grid
	const float x0=mesh->pmin.x, y0=mesh->pmin.y, z0=mesh->pmin.z, x1=mesh->pmax.x, y1=mesh->pmax.y, z1=mesh->pmax.z; // remove all flags in bounding box of mesh
	Kernel kernel_unvoxelize_mesh(device, get_N(), "unvoxelize_mesh", flags, flag, x0, y0, z0, x1, y1, z1);
	kernel_unvoxelize_mesh.run();
}

string LBM_Domain::device_defines(const Device_Info& device_info) const {
	string s;
	s += "\n\t#define def_Nx "+to_string(Nx)+"u";
	s += "\n\t#define def_Ny "+to_string(Ny)+"u";
	s += "\n\t#define def_Nz "+to_string(Nz)+"u";
	s += "\n\t#define def_N "+to_string(get_N())+"ul";
	s += "\n\t#define uxx "+string(get_N()<=(ulong)max_uint ? "uint" : "ulong")+""; // switchable data type for index calculation (32-bit uint / 64-bit ulong)

	s += "\n\t#define def_GNx "+to_string((Nx-2u*(uint)(Dx>1u))*Dx)+"u"; // global LBM grid resolution of all domains together
	s += "\n\t#define def_GNy "+to_string((Ny-2u*(uint)(Dy>1u))*Dy)+"u";
	s += "\n\t#define def_GNz "+to_string((Nz-2u*(uint)(Dz>1u))*Dz)+"u";

	s += "\n\t#define def_Dx "+to_string(Dx)+"u";
	s += "\n\t#define def_Dy "+to_string(Dy)+"u";
	s += "\n\t#define def_Dz "+to_string(Dz)+"u";

	s += "\n\t#define def_Ox "+to_string(Ox)+""; // offsets are signed integer!
	s += "\n\t#define def_Oy "+to_string(Oy)+"";
	s += "\n\t#define def_Oz "+to_string(Oz)+"";

	s += "\n\t#define def_Ax "+to_string(Ny*Nz)+"u";
	s += "\n\t#define def_Ay "+to_string(Nz*Nx)+"u";
	s += "\n\t#define def_Az "+to_string(Nx*Ny)+"u";

	s += "\n\t#define def_domain_offset_x "+to_string(0.5f*(float)((int)Nx+2*Ox+(int)Dx*(2*(int)(Dx>1u)-(int)Nx)))+"f";
	s += "\n\t#define def_domain_offset_y "+to_string(0.5f*(float)((int)Ny+2*Oy+(int)Dy*(2*(int)(Dy>1u)-(int)Ny)))+"f";
	s += "\n\t#define def_domain_offset_z "+to_string(0.5f*(float)((int)Nz+2*Oz+(int)Dz*(2*(int)(Dz>1u)-(int)Nz)))+"f";

	s += "\n\t#define D"+to_string(dims)+"Q"+to_string(velocity_set)+""; // D2Q9/D3Q15/D3Q19/D3Q27
	s += "\n\t#define def_velocity_set "+to_string(velocity_set)+"u"; // LBM velocity set (D2Q9/D3Q15/D3Q19/D3Q27)
	s += "\n\t#define def_dimensions "+to_string(dims)+"u"; // number spatial dimensions (2D or 3D)
	s += "\n\t#define def_transfers "+to_string(trans)+"u"; // number of DDFs that are transferred between multiple domains

	s += "\n\t#define def_c 0.57735027f"; // lattice speed of sound c = 1/sqrt(3)*dt
	s += "\n\t#define def_w " +to_string(1.0f/get_tau())+"f"; // relaxation rate w = dt/tau = dt/(nu/c^2+dt/2) = 1/(3*nu+1/2)

	// velocity set lattice weights
	if(velocity_set==9u) {
		s += "\n\t#define def_w0 (1.0f/2.25f)"; // center (0)
		s += "\n\t#define def_ws (1.0f/9.0f)"; // straight (1-4)
		s += "\n\t#define def_we (1.0f/36.0f)"; // edge (5-8)
	} else if(velocity_set==15u) {
		s += "\n\t#define def_w0 (1.0f/4.5f)"; // center (0)
		s += "\n\t#define def_ws (1.0f/9.0f)"; // straight (1-6)
		s += "\n\t#define def_wc (1.0f/72.0f)"; // corner (7-14)
	} else if(velocity_set==19u) {
		s += "\n\t#define def_w0 (1.0f/3.0f)"; // center (0)
		s += "\n\t#define def_ws (1.0f/18.0f)"; // straight (1-6)
		s += "\n\t#define def_we (1.0f/36.0f)"; // edge (7-18)
	} else if(velocity_set==27u) {
		s += "\n\t#define def_w0 (1.0f/3.375f)"; // center (0)
		s += "\n\t#define def_ws (1.0f/13.5f)"; // straight (1-6)
		s += "\n\t#define def_we (1.0f/54.0f)"; // edge (7-18)
		s += "\n\t#define def_wc (1.0f/216.0f)"; // corner (19-26)
	}

	// collision operator
	if(use_srt) s += "\n\t#define SRT";
	else        s += "\n\t#define TRT";

	s += "\n\t#define TYPE_S 0x01"; // 0b00000001 // (stationary or moving) solid boundary
	s += "\n\t#define TYPE_E 0x02"; // 0b00000010 // equilibrium boundary (inflow/outflow)
	s += "\n\t#define TYPE_T 0x04"; // 0b00000100 // temperature boundary
	s += "\n\t#define TYPE_F 0x08"; // 0b00001000 // fluid
	s += "\n\t#define TYPE_I 0x10"; // 0b00010000 // interface
	s += "\n\t#define TYPE_G 0x20"; // 0b00100000 // gas
	s += "\n\t#define TYPE_X 0x40"; // 0b01000000 // reserved type X
	s += "\n\t#define TYPE_Y 0x80"; // 0b10000000 // reserved type Y

	s += "\n\t#define TYPE_MS 0x03"; // 0b00000011 // cell next to moving solid boundary
	s += "\n\t#define TYPE_BO 0x03"; // 0b00000011 // any flag bit used for boundaries (temperature excluded)
	s += "\n\t#define TYPE_IF 0x18"; // 0b00011000 // change from interface to fluid
	s += "\n\t#define TYPE_IG 0x30"; // 0b00110000 // change from interface to gas
	s += "\n\t#define TYPE_GI 0x38"; // 0b00111000 // change from gas to interface
	s += "\n\t#define TYPE_SU 0x38"; // 0b00111000 // any flag bit used for SURFACE
	s += "\n\t#define TYPE_XY 0xC0"; // 0b11000000 // any flag bit used for X or Y markers

	// FP precision
	if(fp_precision==1) {
		s += "\n\t#define fpxx half"; // switchable data type (scaled IEEE-754 16-bit floating-point format: 1-5-10, exp-30, +-1.99902344, +-1.86446416E-9, +-1.81898936E-12, 3.311 digits)
		s += "\n\t#define fpxx_copy ushort"; // switchable data type for direct copying (scaled IEEE-754 16-bit floating-point format: 1-5-10, exp-30, +-1.99902344, +-1.86446416E-9, +-1.81898936E-12, 3.311 digits)
		s += "\n\t#define load(p,o) (vload_half(o,p)*3.0517578E-5f)"; // special function for loading half
		s += "\n\t#define store(p,o,x) vstore_half_rte((x)*32768.0f,o,p)"; // special function for storing half
	} else if(fp_precision==2) {
		s += "\n\t#define fpxx ushort"; // switchable data type (custom 16-bit floating-point format: 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits), 12.5% slower than IEEE-754 16-bit
		s += "\n\t#define fpxx_copy ushort"; // switchable data type for direct copying (custom 16-bit floating-point format: 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits), 12.5% slower than IEEE-754 16-bit
		s += "\n\t#define load(p,o) half_to_float_custom((p)[o])"; // special function for loading half
		s += "\n\t#define store(p,o,x) (p)[o]=float_to_half_custom(x)"; // special function for storing half
	} else { // FP32
		s += "\n\t#define fpxx float"; // switchable data type (regular 32-bit float)
		s += "\n\t#define fpxx_copy float"; // switchable data type for direct copying (regular 32-bit float)
		s += "\n\t#define load(p,o) (p)[o]"; // regular float read
		s += "\n\t#define store(p,o,x) (p)[o]=(x)"; // regular float write
	}

	// extension flags
	if(has_update_fields)           s += "\n\t#define UPDATE_FIELDS";
	if(has_volume_force)            s += "\n\t#define VOLUME_FORCE";
	if(has_moving_boundaries)       s += "\n\t#define MOVING_BOUNDARIES";
	if(has_equilibrium_boundaries)  s += "\n\t#define EQUILIBRIUM_BOUNDARIES";
	if(has_force_field)             s += "\n\t#define FORCE_FIELD";
	if(has_surface) {
		s += "\n\t#define SURFACE";
		s += "\n\t#define def_6_sigma "+to_string(6.0f*sigma)+"f"; // rho_laplace = 2*o*K, rho = 1-rho_laplace/c^2 = 1-(6*o)*K
	}
	if(has_temperature) {
		s += "\n\t#define TEMPERATURE";
		s += "\n\t#define def_w_T "+to_string(1.0f/(2.0f*alpha+0.5f))+"f"; // wT = dt/tauT = 1/(2*alpha+1/2), alpha = thermal diffusion coefficient
		s += "\n\t#define def_beta "+to_string(beta)+"f"; // thermal expansion coefficient
		s += "\n\t#define def_T_avg "+to_string(T_avg)+"f"; // average temperature
	}
	if(has_subgrid)                 s += "\n\t#define SUBGRID";
	if(has_particles) {
		s += "\n\t#define PARTICLES";
		s += "\n\t#define def_particles_N "+to_string(particles_N)+"ul";
		s += "\n\t#define def_particles_rho "+to_string(particles_rho)+"f";
	}
	return s;
}






vector<Device_Info> smart_device_selection(const uint D) {
	const vector<Device_Info>& devices = get_devices(); // a vector of all available OpenCL devices
	vector<Device_Info> device_infos(D);
	const int user_specified_devices = (int)main_arguments.size();
	if(user_specified_devices>0) { // user has selevted specific devices as command line arguments
		if(user_specified_devices==D) { // as much specified devices as domains
			for(uint d=0; d<D; d++) device_infos[d] = select_device_with_id(to_uint(main_arguments[d]), devices); // use list of devices IDs specified by user
		} else {
			print_warning("Incorrect number of devices specified. Using single fastest device for all domains.");
			for(uint d=0; d<D; d++) device_infos[d] = select_device_with_most_flops(devices);
		}
	} else { // device auto-selection
		vector<vector<Device_Info>> device_type_ids; // a vector of all different devices, containing vectors of their device IDs
		for(uint i=0u; i<(uint)devices.size(); i++) {
			const string name_i = devices[i].name;
			bool already_exists = false;
			for(uint j=0u; j<(uint)device_type_ids.size(); j++) {
				const string name_j = device_type_ids[j][0].name;
				if(name_i==name_j) {
					device_type_ids[j].push_back(devices[i]);
					already_exists = true;
				}
			}
			if(!already_exists) device_type_ids.push_back(vector<Device_Info>(1, devices[i]));
		}
		float best_value = -1.0f;
		int best_j = -1;
		for(uint j=0u; j<(uint)device_type_ids.size(); j++) {
			const float value = device_type_ids[j][0].tflops;
			if((uint)device_type_ids[j].size()>=D && value>best_value) {
				best_value = value;
				best_j = j;
			}
		}
		if(best_j>=0) { // select all devices of fastest device type with at least D devices of the same type
			for(uint d=0; d<D; d++) device_infos[d] = device_type_ids[best_j][d];
		} else {
			print_warning("Not enough devices of the same type available. Using single fastest device for all domains.");
			for(uint d=0; d<D; d++) device_infos[d] = select_device_with_most_flops(devices);
		}
		//for(uint j=0u; j<(uint)device_type_ids.size(); j++) print_info("Device Type "+to_string(j)+" ("+device_type_ids[j][0].name+"): "+to_string((uint)device_type_ids[j].size())+"x");
	}
	return device_infos;
}

LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) // single device
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho)
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, fx, fy, fz, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const float nu, const uint particles_N, const float particles_rho)
	:LBM(Nx, Ny, Nz, 1u, 1u, 1u, nu, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, Dx, Dy, Dz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) // single device
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint3 N, const float nu, const float fx, const float fy, const float fz, const uint particles_N, const float particles_rho)
	:LBM(N.x, N.y, N.z, 1u, 1u, 1u, nu, fx, fy, fz, 0.0f, 0.0f, 0.0f, particles_N, particles_rho) { // delegating constructor
}
LBM::LBM(const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho) { // multiple devices
	const uint NDx=(Nx/Dx)*Dx, NDy=(Ny/Dy)*Dy, NDz=(Nz/Dz)*Dz; // make resolution equally divisible by domains
	if(NDx!=Nx||NDy!=Ny||NDz!=Nz) print_warning("LBM grid ("+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+") is not equally divisible in domains ("+to_string(Dx)+"x"+to_string(Dy)+"x"+to_string(Dz)+"). Changing resolution to ("+to_string(NDx)+"x"+to_string(NDy)+"x"+to_string(NDz)+").");
	this->Nx = NDx; this->Ny = NDy; this->Nz = NDz;
	this->Dx = Dx; this->Dy = Dy; this->Dz = Dz;
	const uint D = Dx*Dy*Dz;
	const uint Hx=Dx>1u, Hy=Dy>1u, Hz=Dz>1u; // halo offsets
	const vector<Device_Info>& device_infos = smart_device_selection(D);
	sanity_checks_constructor(device_infos, this->Nx, this->Ny, this->Nz, Dx, Dy, Dz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	lbm_domain = new LBM_Domain*[D];
	for(uint d=0u; d<D; d++) { // parallel_for((ulong)D, D, [&](ulong d) {
		const uint x=((uint)d%(Dx*Dy))%Dx, y=((uint)d%(Dx*Dy))/Dx, z=(uint)d/(Dx*Dy); // d = x+(y+z*Dy)*Dx
		lbm_domain[d] = new LBM_Domain(device_infos[d], this->Nx/Dx+2u*Hx, this->Ny/Dy+2u*Hy, this->Nz/Dz+2u*Hz, Dx, Dy, Dz, (int)(x*this->Nx/Dx)-(int)Hx, (int)(y*this->Ny/Dy)-(int)Hy, (int)(z*this->Nz/Dz)-(int)Hz, nu, fx, fy, fz, sigma, alpha, beta, particles_N, particles_rho);
	} // });
	{
		Memory<float>** buffers_rho = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_rho[d] = &(lbm_domain[d]->rho);
		rho = Memory_Container(this, buffers_rho, "rho");
	} {
		Memory<float>** buffers_u = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_u[d] = &(lbm_domain[d]->u);
		u = Memory_Container(this, buffers_u, "u");
	} {
		Memory<uchar>** buffers_flags = new Memory<uchar>*[D];
		for(uint d=0u; d<D; d++) buffers_flags[d] = &(lbm_domain[d]->flags);
		flags = Memory_Container(this, buffers_flags, "flags");
	} {
#ifdef FORCE_FIELD
		Memory<float>** buffers_F = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_F[d] = &(lbm_domain[d]->F);
		F = Memory_Container(this, buffers_F, "F");
#endif // FORCE_FIELD
	} {
#ifdef SURFACE
		Memory<float>** buffers_phi = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_phi[d] = &(lbm_domain[d]->phi);
		phi = Memory_Container(this, buffers_phi, "phi");
#endif // SURFACE
	} {
#ifdef TEMPERATURE
		Memory<float>** buffers_T = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_T[d] = &(lbm_domain[d]->T);
		T = Memory_Container(this, buffers_T, "T");
#endif // TEMPERATURE
	} {
#ifdef PARTICLES
		particles = &(lbm_domain[0]->particles);
#endif // PARTICLES
	}
}
LBM::LBM(const SimulationConfig& config) { // runtime config constructor
	const uint NDx=(config.Nx/config.Dx)*config.Dx, NDy=(config.Ny/config.Dy)*config.Dy, NDz=(config.Nz/config.Dz)*config.Dz; // make resolution equally divisible by domains
	if(NDx!=config.Nx||NDy!=config.Ny||NDz!=config.Nz) print_warning("LBM grid ("+to_string(config.Nx)+"x"+to_string(config.Ny)+"x"+to_string(config.Nz)+") is not equally divisible in domains ("+to_string(config.Dx)+"x"+to_string(config.Dy)+"x"+to_string(config.Dz)+"). Changing resolution to ("+to_string(NDx)+"x"+to_string(NDy)+"x"+to_string(NDz)+").");
	this->Nx = NDx; this->Ny = NDy; this->Nz = NDz;
	this->Dx = config.Dx; this->Dy = config.Dy; this->Dz = config.Dz;
	const uint D = config.Dx*config.Dy*config.Dz;
	const uint Hx=config.Dx>1u, Hy=config.Dy>1u, Hz=config.Dz>1u; // halo offsets
	const vector<Device_Info>& device_infos = smart_device_selection(D);
	sanity_checks_constructor(device_infos, this->Nx, this->Ny, this->Nz, config.Dx, config.Dy, config.Dz, config.nu, config.fx, config.fy, config.fz, config.sigma, config.alpha, config.beta, config.particles_N, config.particles_rho, &config);
	lbm_domain = new LBM_Domain*[D];
	for(uint d=0u; d<D; d++) { // parallel_for((ulong)D, D, [&](ulong d) {
		const uint x=((uint)d%(config.Dx*config.Dy))%config.Dx, y=((uint)d%(config.Dx*config.Dy))/config.Dx, z=(uint)d/(config.Dx*config.Dy); // d = x+(y+z*Dy)*Dx
		lbm_domain[d] = new LBM_Domain(device_infos[d], config, this->Nx/config.Dx+2u*Hx, this->Ny/config.Dy+2u*Hy, this->Nz/config.Dz+2u*Hz, config.Dx, config.Dy, config.Dz, (int)(x*this->Nx/config.Dx)-(int)Hx, (int)(y*this->Ny/config.Dy)-(int)Hy, (int)(z*this->Nz/config.Dz)-(int)Hz);
	} // });
	{
		Memory<float>** buffers_rho = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_rho[d] = &(lbm_domain[d]->rho);
		rho = Memory_Container(this, buffers_rho, "rho");
	} {
		Memory<float>** buffers_u = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_u[d] = &(lbm_domain[d]->u);
		u = Memory_Container(this, buffers_u, "u");
	} {
		Memory<uchar>** buffers_flags = new Memory<uchar>*[D];
		for(uint d=0u; d<D; d++) buffers_flags[d] = &(lbm_domain[d]->flags);
		flags = Memory_Container(this, buffers_flags, "flags");
	} {
#ifdef FORCE_FIELD
		Memory<float>** buffers_F = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_F[d] = &(lbm_domain[d]->F);
		F = Memory_Container(this, buffers_F, "F");
#endif // FORCE_FIELD
	} {
#ifdef SURFACE
		Memory<float>** buffers_phi = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_phi[d] = &(lbm_domain[d]->phi);
		phi = Memory_Container(this, buffers_phi, "phi");
#endif // SURFACE
	} {
#ifdef TEMPERATURE
		Memory<float>** buffers_T = new Memory<float>*[D];
		for(uint d=0u; d<D; d++) buffers_T[d] = &(lbm_domain[d]->T);
		T = Memory_Container(this, buffers_T, "T");
#endif // TEMPERATURE
	} {
#ifdef PARTICLES
		particles = &(lbm_domain[0]->particles);
#endif // PARTICLES
	}
}
LBM::~LBM() {
	info.print_finalize();
	for(uint d=0u; d<get_D(); d++) delete lbm_domain[d];
	delete[] lbm_domain;
}

void LBM::sanity_checks_constructor(const vector<Device_Info>& device_infos, const uint Nx, const uint Ny, const uint Nz, const uint Dx, const uint Dy, const uint Dz, const float nu, const float fx, const float fy, const float fz, const float sigma, const float alpha, const float beta, const uint particles_N, const float particles_rho, const SimulationConfig* cfg) { // sanity checks on grid resolution and extension support
	if((ulong)Nx*(ulong)Ny*(ulong)Nz==0ull) print_error("Grid point number is 0: "+to_string(Nx)+"x"+to_string(Ny)+"x"+to_string(Nz)+" = 0.");
	if(Dx*Dy*Dz==0u) print_error("You specified 0 LBM grid domains ("+to_string(Dx)+"x"+to_string(Dy)+"x"+to_string(Dz)+"). There has to be at least 1 domain in every direction. Check your input in LBM constructor.");
	const uint local_Nx=Nx/Dx+2u*(Dx>1u), local_Ny=Ny/Dy+2u*(Dy>1u), local_Nz=Nz/Dz+2u*(Dz>1u);
	uint memory_available = max_uint; // in MB
	for(Device_Info device_info : device_infos) memory_available = min(memory_available, device_info.memory);
	uint memory_required = (uint)((ulong)Nx*(ulong)Ny*(ulong)Nz/((ulong)(Dx*Dy*Dz))*(ulong)bytes_per_cell_device()/1048576ull); // in MB
	if(memory_required>memory_available) {
		float factor = cbrt((float)memory_available/(float)memory_required);
		const uint maxNx=(uint)(factor*(float)Nx), maxNy=(uint)(factor*(float)Ny), maxNz=(uint)(factor*(float)Nz);
		string message = "Grid resolution ("+to_string(Nx)+", "+to_string(Ny)+", "+to_string(Nz)+") is too large: "+to_string(Dx*Dy*Dz)+"x "+to_string(memory_required)+" MB required, "+to_string(Dx*Dy*Dz)+"x "+to_string(memory_available)+" MB available. Largest possible resolution is ("+to_string(maxNx)+", "+to_string(maxNy)+", "+to_string(maxNz)+"). Restart the simulation with lower resolution or on different device(s) with more memory.";
#if !defined(FP16S)&&!defined(FP16C)
		uint memory_required_fp16 = (uint)((ulong)Nx*(ulong)Ny*(ulong)Nz/((ulong)(Dx*Dy*Dz))*(ulong)(bytes_per_cell_device()-velocity_set*2u)/1048576ull); // in MB
		float factor_fp16 = cbrt((float)memory_available/(float)memory_required_fp16);
		const uint maxNx_fp16=(uint)(factor_fp16*(float)Nx), maxNy_fp16=(uint)(factor_fp16*(float)Ny), maxNz_fp16=(uint)(factor_fp16*(float)Nz);
		message += " Consider using FP16S/FP16C memory compression to double maximum grid resolution to a maximum of ("+to_string(maxNx_fp16)+", "+to_string(maxNy_fp16)+", "+to_string(maxNz_fp16)+"); for this, uncomment \"#define FP16S\" or \"#define FP16C\" in defines.hpp.";
#endif // !FP16S&&!FP16C
		print_error(message);
	}
	if(nu==0.0f) print_error("Viscosity cannot be 0. Change it in setup.cpp."); // sanity checks for viscosity
	else if(nu<0.0f) print_error("Viscosity cannot be negative. Remove the \"-\" in setup.cpp.");
#ifdef D2Q9
	if(Nz!=1u) print_error("D2Q9 is the 2D velocity set. You have to set Nz=1u in the LBM constructor! Currently you have set Nz="+to_string(Nz)+"u.");
#endif // D2Q9
#if !defined(SRT)&&!defined(TRT)
	print_error("No LBM collision operator selected. Uncomment either \"#define SRT\" or \"#define TRT\" in defines.hpp");
#elif defined(SRT)&&defined(TRT)
	print_error("Too many LBM collision operators selected. Comment out either \"#define SRT\" or \"#define TRT\" in defines.hpp");
#endif // SRT && TRT
	// Extension sanity checks - use runtime config if available, else compile-time defines
	bool vol_ok = cfg ? cfg->volume_force :
#ifdef VOLUME_FORCE
		true;
#else
		false;
#endif
	if(!vol_ok && (fx!=0.0f||fy!=0.0f||fz!=0.0f))
		print_error("Volume force is set in LBM constructor but VOLUME_FORCE is not enabled.");
	bool surf_ok = cfg ? cfg->surface :
#ifdef SURFACE
		true;
#else
		false;
#endif
	if(!surf_ok && sigma!=0.0f)
		print_error("Surface tension is set in LBM constructor but SURFACE is not enabled.");
	bool temp_ok = cfg ? cfg->temperature :
#ifdef TEMPERATURE
		true;
#else
		false;
#endif
	if(!temp_ok && (alpha!=0.0f||beta!=0.0f))
		print_error("Thermal diffusion/expansion coefficients are set but TEMPERATURE is not enabled.");
	bool part_ok = cfg ? cfg->particles :
#ifdef PARTICLES
		true;
#else
		false;
#endif
	if(part_ok && particles_N==0u)
		print_error("The PARTICLES extension is enabled but the number of particles is set to 0.");
	if(!part_ok && particles_N>0u)
		print_error("The PARTICLES extension is disabled but the number of particles is set to "+to_string(particles_N)+">0.");
	bool ff_ok = cfg ? cfg->force_field :
#ifdef FORCE_FIELD
		true;
#else
		false;
#endif
	if(part_ok && !vol_ok && !ff_ok && particles_rho!=1.0f)
		print_error("Particle density is set unequal to 1, but 2-way-coupling is not enabled.");
}

void LBM::sanity_checks_initialization() { // sanity checks during initialization on used extensions based on used flags
	uchar flags_used = 0u;
	bool moving_boundaries_used=false, equilibrium_boundaries_used=false, surface_used=false, temperature_used=false; // identify used extensions based used flags
	const uint threads = thread::hardware_concurrency();
	vector<uchar> t_flags_used(threads, 0u);
	vector<char> t_moving_boundaries_used(threads, false); // don't use vector<bool> as it uses bit-packing which is broken for multithreading
	vector<char> t_equilibrium_boundaries_used(threads, false); // don't use vector<bool> as it uses bit-packing which is broken for multithreading
	parallel_for(get_N(), threads, [&](ulong n, uint t) {
		const uchar flagsn = flags[n];
		const uchar flagsn_bo = flagsn&(TYPE_S|TYPE_E);
		t_flags_used[t] = t_flags_used[t]|flagsn;
		if(flagsn_bo&TYPE_S) t_moving_boundaries_used[t] = t_moving_boundaries_used[t] || (((flagsn_bo==TYPE_S)&&(u.x[n]!=0.0f||u.y[n]!=0.0f||u.z[n]!=0.0f))||(flagsn_bo==(TYPE_S|TYPE_E)));
		t_equilibrium_boundaries_used[t] = t_equilibrium_boundaries_used[t] || flagsn_bo==TYPE_E;
	});
	for(uint t=0u; t<threads; t++) {
		flags_used = flags_used|t_flags_used[t];
		moving_boundaries_used = moving_boundaries_used || t_moving_boundaries_used[t];
		equilibrium_boundaries_used = equilibrium_boundaries_used || t_equilibrium_boundaries_used[t];
	}
	surface_used = (bool)(flags_used&(TYPE_F|TYPE_I|TYPE_G));
	temperature_used = (bool)(flags_used&TYPE_T);
#ifndef MOVING_BOUNDARIES
	if(moving_boundaries_used) print_warning("Some boundary cells have non-zero velocity, but MOVING_BOUNDARIES is not enabled. If you intend to use moving boundaries, uncomment \"#define MOVING_BOUNDARIES\" in defines.hpp.");
#else // MOVING_BOUNDARIES
	if(!moving_boundaries_used) print_warning("The MOVING_BOUNDARIES extension is enabled but no moving boundary cells (TYPE_S flag and velocity unequal to zero) are placed in the simulation box. You may disable the extension by commenting out \"#define MOVING_BOUNDARIES\" in defines.hpp.");
#endif // MOVING_BOUNDARIES
#ifndef EQUILIBRIUM_BOUNDARIES
	if(equilibrium_boundaries_used) print_error("Some cells are set as equilibrium boundaries with the TYPE_E flag, but EQUILIBRIUM_BOUNDARIES is not enabled. Uncomment \"#define EQUILIBRIUM_BOUNDARIES\" in defines.hpp.");
#else // EQUILIBRIUM_BOUNDARIES
	if(!equilibrium_boundaries_used) print_warning("The EQUILIBRIUM_BOUNDARIES extension is enabled but no equilibrium boundary cells (TYPE_E flag) are placed in the simulation box. You may disable the extension by commenting out \"#define EQUILIBRIUM_BOUNDARIES\" in defines.hpp.");
#endif // EQUILIBRIUM_BOUNDARIES
#ifndef SURFACE
	if(surface_used) print_error("Some cells are set as fluid/interface/gas with the TYPE_F/TYPE_I/TYPE_G flags, but SURFACE is not enabled. Uncomment \"#define SURFACE\" in defines.hpp.");
#else // SURFACE
	if(!surface_used) print_error("The SURFACE extension is enabled but no fluid/interface/gas cells (TYPE_F/TYPE_I/TYPE_G flags) are placed in the simulation box. Disable the extension by commenting out \"#define SURFACE\" in defines.hpp.");
#endif // SURFACE
#ifndef TEMPERATURE
	if(temperature_used) print_error("Some cells are set as temperature boundary with the TYPE_T flag, but TEMPERATURE is not enabled. Uncomment \"#define TEMPERATURE\" in defines.hpp.");
#endif // TEMPERATURE
}

void LBM::initialize() { // write all data fields to device and call kernel_initialize
#ifndef BENCHMARK
	sanity_checks_initialization();
#endif // BENCHMARK

	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->rho.enqueue_write_to_device();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->u.enqueue_write_to_device();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->flags.enqueue_write_to_device();
#ifdef FORCE_FIELD
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->F.enqueue_write_to_device();
	communicate_F();
#endif // FORCE_FIELD
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->phi.enqueue_write_to_device();
#endif // SURFACE
#ifdef TEMPERATURE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->T.enqueue_write_to_device();
#endif // TEMPERATURE
#ifdef PARTICLES
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->particles.enqueue_write_to_device();
	communicate_particles();
#endif // PARTICLES

	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step(); // the communicate calls at initialization need an odd time step
	communicate_rho_u_flags();
#ifdef SURFACE
	communicate_phi_massex_flags();
#endif // SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_initialize(); // odd time step is baked-in the kernel
	communicate_rho_u_flags();
#ifdef SURFACE
	communicate_phi_massex_flags();
#endif // SURFACE
	communicate_fi(); // time step must be odd here
#ifdef TEMPERATURE
	communicate_T(); // T halo data is required for field_slice rendering
	communicate_gi(); // time step must be odd here
#endif // TEMPERATURE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->reset_time_step(); // set time step to 0 again
	initialized = true;
}

void LBM::do_time_step() { // call kernel_stream_collide to perform one LBM time step
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_0();
#endif // SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_stream_collide(); // run LBM stream_collide kernel after domain communication
#ifdef SURFACE
	communicate_rho_u_flags(); // rho/u/flags halo data is required for SURFACE extension
#endif // SURFACE
#ifdef SURFACE
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_1();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_2();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_surface_3();
	communicate_phi_massex_flags();
#endif // SURFACE
	communicate_fi();
#ifdef TEMPERATURE
	communicate_gi();
#endif // TEMPERATURE
#ifdef PARTICLES
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_integrate_particles(); // intgegrate particles forward in time and couple particles to fluid
	communicate_particles(); // communicate_F() is not required in do_time_step()
#endif // PARTICLES
	if(get_D()==1u) for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // this additional domain synchronization barrier is only required in single-GPU, as communication calls already provide all necessary synchronization barriers in multi-GPU
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step();
}

void LBM::run(const ulong steps, const ulong total_steps) { // initializes the LBM simulation (copies data to device and runs initialize kernel), then runs LBM
	info.append(steps, total_steps, get_t()); // total_steps parameter is just for runtime estimation
	if(!initialized) {
		initialize();
		info.print_initialize(this); // only print setup info if the setup is new (run() was not called before)
	}
	Clock clock;
	for(ulong i=1ull; i<=steps; i++) {
		clock.start();
		do_time_step();
		info.update(clock.stop());
	}
	if(get_D()>1u) for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // wait for everything to finish (multi-GPU only)
}

void LBM::update_fields() { // update fields (rho, u, T) manually
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_fields();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}

void LBM::reset() { // reset simulation (takes effect in following run() call)
	initialized = false;
}

#ifdef FORCE_FIELD
void LBM::update_force_field() { // calculate forces from fluid on TYPE_S cells
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_force_field();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}
float3 LBM::object_center_of_mass(const uchar flag_marker) { // calculate center of mass of all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_center_of_mass(flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_com = float3(0.0f, 0.0f, 0.0f);
	ulong object_cells = 0ull;
	for(uint d=0u; d<get_D(); d++) {
		object_com += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
		object_cells += (ulong)as_uint(lbm_domain[d]->object_sum.w[0]);
	}
	return object_com/(float)object_cells;
}
float3 LBM::object_force(const uchar flag_marker) { // add up force for all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_force(flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_force = float3(0.0f, 0.0f, 0.0f);
	for(uint d=0u; d<get_D(); d++) object_force += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
	return object_force;
}
float3 LBM::object_torque(const float3& rotation_center, const uchar flag_marker) { // add up torque around specified rotation center for all cells flagged with flag_marker
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_object_torque(rotation_center, flag_marker);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
	float3 object_torque = float3(0.0f, 0.0f, 0.0f);
	for(uint d=0u; d<get_D(); d++) object_torque += float3(lbm_domain[d]->object_sum.x[0], lbm_domain[d]->object_sum.y[0], lbm_domain[d]->object_sum.z[0]);
	return object_torque;
}
#endif // FORCE_FIELD

#ifdef MOVING_BOUNDARIES
void LBM::update_moving_boundaries() { // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_update_moving_boundaries();
	communicate_flags();
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}
#endif // MOVING_BOUNDARIES

#if defined(PARTICLES)&&!defined(FORCE_FIELD)
void LBM::integrate_particles(const ulong steps, const ulong total_steps, const uint time_step_multiplicator) { // intgegrate passive tracer particles forward in time in stationary flow field
	info.append(steps, total_steps, get_t());
	Clock clock;
	for(ulong i=1ull; i<=steps; i+=(ulong)time_step_multiplicator) {
		clock.start();
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_integrate_particles(time_step_multiplicator);
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->increment_time_step(time_step_multiplicator);
		info.update(clock.stop());
	}
}
#endif // PARTICLES&&!FORCE_FIELD

void LBM::write_status(const string& path) { // write LBM status report to a .txt file
	string status = "";
	status += "Grid Resolution = "+to_string(Nx)+" x "+to_string(Ny)+" x "+to_string(Nz)+" = "+to_string(get_N())+"\n";
	status += "Grid Domains = "+to_string(Dx)+" x "+to_string(Dy)+" x "+to_string(Dz)+" = "+to_string(get_D())+"\n";
	status += "LBM Type = D"+string(get_velocity_set()==9 ? "2" : "3")+"Q"+to_string(get_velocity_set())+" "+info.collision+"\n";
	status += "Memory Usage = CPU "+to_string(info.cpu_mem_required)+" MB, GPU "+to_string(get_D())+"x "+to_string(info.gpu_mem_required)+" MB\n";
	status += "Maximum Allocation Size = "+to_string((uint)(get_N()/(ulong)get_D()*(ulong)(get_velocity_set()*sizeof(fpxx))/1048576ull))+" MB\n";
	status += "Time Steps = "+to_string(get_t())+" / "+(info.steps==max_ulong ? "infinite" : to_string(info.steps))+"\n";
	status += "Runtime = "+print_time(info.runtime_total)+" (total) = "+print_time(info.runtime_lbm)+" (LBM) + "+print_time(info.runtime_total-info.runtime_lbm)+" (rendering and data evaluation)\n";
	status += "Average MLUPs/s = "+to_string(to_uint(1E-6*(double)get_N()*(double)get_t()/info.runtime_lbm))+"\n";
	status += "Kinematic Viscosity = "+to_string(get_nu())+"\n";
	status += "Relaxation Time = "+to_string(get_tau())+"\n";
	status += "Maximum Reynolds Number = "+to_string(get_Re_max())+"\n";
#ifdef VOLUME_FORCE
	status += "Volume Force = ("+to_string(get_fx())+", "+to_string(get_fy())+", "+to_string(get_fz())+")\n";
#endif // VOLUME_FORCE
#ifdef SURFACE
	status += "Surface Tension Coefficient = "+to_string(get_sigma())+"\n";
#endif // SURFACE
#ifdef TEMPERATURE
	status += "Thermal Diffusion Coefficient = "+to_string(get_alpha())+"\n";
	status += "Thermal Expansion Coefficient = "+to_string(get_beta())+"\n";
#endif // TEMPERATURE
	const string filename = default_filename(path, "status", ".txt", get_t());
	write_file(filename, status);
}

void LBM::voxelize_mesh_on_device(const Mesh* mesh, const uchar flag, const float3& rotation_center, const float3& linear_velocity, const float3& rotational_velocity) { // voxelize triangle mesh
	if(get_D()==1u) {
		lbm_domain[0]->voxelize_mesh_on_device(mesh, flag, rotation_center, linear_velocity, rotational_velocity); // if this crashes on Windows, create a TdrDelay 32-bit DWORD with decimal value 300 in Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
	} else {
		parallel_for(get_D(), get_D(), [&](uint d) {
			lbm_domain[d]->voxelize_mesh_on_device(mesh, flag, rotation_center, linear_velocity, rotational_velocity);
		});
	}
#ifdef MOVING_BOUNDARIES
	if((flag&(TYPE_S|TYPE_E))==TYPE_S&&(length(linear_velocity)>0.0f||length(rotational_velocity)>0.0f)) update_moving_boundaries();
#endif // MOVING_BOUNDARIES
	if(!initialized) {
		flags.read_from_device();
		u.read_from_device();
	}
}
void LBM::unvoxelize_mesh_on_device(const Mesh* mesh, const uchar flag) { // remove voxelized triangle mesh from LBM grid by removing all flags in mesh bounding box (only required when bounding box size changes during re-voxelization)
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_unvoxelize_mesh_on_device(mesh, flag);
	for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue();
}
void LBM::write_mesh_to_vtk(const Mesh* mesh, const string& path, const bool convert_to_si_units) const { // write mesh to binary .vtk file
	const string filename = default_filename(path, "mesh", ".vtk", get_t());
	const string header_1 = "# vtk DataFile Version 3.0\nFluidX3D "+filename.substr(filename.rfind('/')+1)+"\nBINARY\nDATASET POLYDATA\nPOINTS "+to_string(3u*mesh->triangle_number)+" float\n";
	const string header_2 = "POLYGONS "+to_string(mesh->triangle_number)+" "+to_string(4u*mesh->triangle_number)+"\n";
	float* points = new float[9u*mesh->triangle_number];
	int* triangles = new int[4u*mesh->triangle_number];
	const float spacing = convert_to_si_units ? units.si_x(1.0f) : 1.0f;
	const float3 offset = center();
	parallel_for(mesh->triangle_number, [&](uint i) {
		points[9u*i   ] = reverse_bytes(spacing*(mesh->p0[i].x-offset.x));
		points[9u*i+1u] = reverse_bytes(spacing*(mesh->p0[i].y-offset.y));
		points[9u*i+2u] = reverse_bytes(spacing*(mesh->p0[i].z-offset.z));
		points[9u*i+3u] = reverse_bytes(spacing*(mesh->p1[i].x-offset.x));
		points[9u*i+4u] = reverse_bytes(spacing*(mesh->p1[i].y-offset.y));
		points[9u*i+5u] = reverse_bytes(spacing*(mesh->p1[i].z-offset.z));
		points[9u*i+6u] = reverse_bytes(spacing*(mesh->p2[i].x-offset.x));
		points[9u*i+7u] = reverse_bytes(spacing*(mesh->p2[i].y-offset.y));
		points[9u*i+8u] = reverse_bytes(spacing*(mesh->p2[i].z-offset.z));
		triangles[4u*i   ] = reverse_bytes(3); // 3 vertices per triangle
		triangles[4u*i+1u] = reverse_bytes(3*(int)i  ); // vertex 0
		triangles[4u*i+2u] = reverse_bytes(3*(int)i+1); // vertex 1
		triangles[4u*i+3u] = reverse_bytes(3*(int)i+2); // vertex 2
	});
	create_folder(filename);
	std::ofstream file(filename, std::ios::out|std::ios::binary);
	file.write(header_1.c_str(), header_1.length()); // write non-binary file header
	file.write((char*)points, 4u*9u*mesh->triangle_number); // write binary data
	file.write(header_2.c_str(), header_2.length()); // write non-binary file header
	file.write((char*)triangles, 4u*4u*mesh->triangle_number); // write binary data
	file.close();
	delete[] points;
	delete[] triangles;
	info.allow_printing.lock();
	print_info("File \""+filename+"\" saved.");
	info.allow_printing.unlock();
}
void LBM::voxelize_stl(const string& path, const float3& center, const float3x3& rotation, const float size, const uchar flag) { // voxelize triangle mesh
	const Mesh* mesh = read_stl(path, this->size(), center, rotation, size);
	flags.write_to_device();
	voxelize_mesh_on_device(mesh, flag);
	delete mesh;
	flags.read_from_device();
}
void LBM::voxelize_stl(const string& path, const float3x3& rotation, const float size, const uchar flag) { // read and voxelize binary .stl file (place in box center)
	voxelize_stl(path, center(), rotation, size, flag);
}
void LBM::voxelize_stl(const string& path, const float3& center, const float size, const uchar flag) { // read and voxelize binary .stl file (no rotation)
	voxelize_stl(path, center, float3x3(1.0f), size, flag);
}
void LBM::voxelize_stl(const string& path, const float size, const uchar flag) { // read and voxelize binary .stl file (place in box center, no rotation)
	voxelize_stl(path, center(), float3x3(1.0f), size, flag);
}






void LBM_Domain::allocate_transfer(Device& device) { // allocate all memory for multi-device trqansfer
	ulong Amax = 0ull; // maximum domain side area of communicated directions
	if(Dx>1u) Amax = max(Amax, (ulong)Ny*(ulong)Nz); // Ax
	if(Dy>1u) Amax = max(Amax, (ulong)Nz*(ulong)Nx); // Ay
	if(Dz>1u) Amax = max(Amax, (ulong)Nx*(ulong)Ny); // Az

	transfer_buffer_p = Memory<char>(device, Amax, max(trans*(uint)sizeof(fpxx), 17u), true, true, 0, false); // only allocate one set of transfer buffers in plus/minus directions, for all x/y/z transfers
	transfer_buffer_m = Memory<char>(device, Amax, max(trans*(uint)sizeof(fpxx), 17u), true, true, 0, false); // these transfer buffers must not be zero-copy!

	kernel_transfer[enum_transfer_field::fi              ][0] = Kernel(device, 0ull, "transfer_extract_fi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fi);
	kernel_transfer[enum_transfer_field::fi              ][1] = Kernel(device, 0ull, "transfer__insert_fi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, fi);
	kernel_transfer[enum_transfer_field::rho_u_flags     ][0] = Kernel(device, 0ull, "transfer_extract_rho_u_flags"     , 0u, t, transfer_buffer_p, transfer_buffer_m, rho, u, flags);
	kernel_transfer[enum_transfer_field::rho_u_flags     ][1] = Kernel(device, 0ull, "transfer__insert_rho_u_flags"     , 0u, t, transfer_buffer_p, transfer_buffer_m, rho, u, flags);
	kernel_transfer[enum_transfer_field::flags           ][0] = Kernel(device, 0ull, "transfer_extract_flags"           , 0u, t, transfer_buffer_p, transfer_buffer_m, flags);
	kernel_transfer[enum_transfer_field::flags           ][1] = Kernel(device, 0ull, "transfer__insert_flags"           , 0u, t, transfer_buffer_p, transfer_buffer_m, flags);
#ifdef FORCE_FIELD
	kernel_transfer[enum_transfer_field::F               ][0] = Kernel(device, 0ull, "transfer_extract_F"               , 0u, t, transfer_buffer_p, transfer_buffer_m, F);
	kernel_transfer[enum_transfer_field::F               ][1] = Kernel(device, 0ull, "transfer__insert_F"               , 0u, t, transfer_buffer_p, transfer_buffer_m, F);
#endif // FORCE_FIELD
#ifdef SURFACE
	kernel_transfer[enum_transfer_field::phi_massex_flags][0] = Kernel(device, 0ull, "transfer_extract_phi_massex_flags", 0u, t, transfer_buffer_p, transfer_buffer_m, phi, massex, flags);
	kernel_transfer[enum_transfer_field::phi_massex_flags][1] = Kernel(device, 0ull, "transfer__insert_phi_massex_flags", 0u, t, transfer_buffer_p, transfer_buffer_m, phi, massex, flags);
#endif // SURFACE
#ifdef TEMPERATURE
	kernel_transfer[enum_transfer_field::gi              ][0] = Kernel(device, 0ull, "transfer_extract_gi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, gi);
	kernel_transfer[enum_transfer_field::gi              ][1] = Kernel(device, 0ull, "transfer__insert_gi"              , 0u, t, transfer_buffer_p, transfer_buffer_m, gi);
	kernel_transfer[enum_transfer_field::T               ][0] = Kernel(device, 0ull, "transfer_extract_T"               , 0u, t, transfer_buffer_p, transfer_buffer_m, T);
	kernel_transfer[enum_transfer_field::T               ][1] = Kernel(device, 0ull, "transfer__insert_T"               , 0u, t, transfer_buffer_p, transfer_buffer_m, T);
#endif // TEMPERATURE
}

ulong LBM_Domain::get_area(const uint direction) {
	const ulong A[3] = { (ulong)Ny*(ulong)Nz, (ulong)Nz*(ulong)Nx, (ulong)Nx*(ulong)Ny };
	return A[direction];
}
void LBM_Domain::enqueue_transfer_extract_field(Kernel& kernel_transfer_extract_field, const uint direction, const uint bytes_per_cell) {
	kernel_transfer_extract_field.set_ranges(get_area(direction)); // direction: x=0, y=1, z=2
	kernel_transfer_extract_field.set_parameters(0u, direction, get_t()).enqueue_run(); // selective in-VRAM copy
	transfer_buffer_p.enqueue_read_from_device(0ull, kernel_transfer_extract_field.range()*(ulong)bytes_per_cell); // PCIe copy (+)
	transfer_buffer_m.enqueue_read_from_device(0ull, kernel_transfer_extract_field.range()*(ulong)bytes_per_cell); // PCIe copy (-)
}
void LBM_Domain::enqueue_transfer_insert_field(Kernel& kernel_transfer_insert_field, const uint direction, const uint bytes_per_cell) {
	kernel_transfer_insert_field.set_ranges(get_area(direction)); // direction: x=0, y=1, z=2
	transfer_buffer_p.enqueue_write_to_device(0ull, kernel_transfer_insert_field.range()*(ulong)bytes_per_cell); // PCIe copy (+)
	transfer_buffer_m.enqueue_write_to_device(0ull, kernel_transfer_insert_field.range()*(ulong)bytes_per_cell); // PCIe copy (-)
	kernel_transfer_insert_field.set_parameters(0u, direction, get_t()).enqueue_run(); // selective in-VRAM copy
}
void LBM::communicate_field(const enum_transfer_field field, const uint bytes_per_cell) {
	if(Dx>1u) { // communicate in x-direction
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_transfer_extract_field(lbm_domain[d]->kernel_transfer[field][0], 0u, bytes_per_cell); // selective in-VRAM copy (x) + PCIe copy
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
		for(uint d=0u; d<get_D(); d++) {
			const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy), dxp=((x+1u)%Dx)+(y+z*Dy)*Dx; // d = x+(y+z*Dy)*Dx
			lbm_domain[d]->transfer_buffer_p.exchange_host_buffer(lbm_domain[dxp]->transfer_buffer_m.exchange_host_buffer(lbm_domain[d]->transfer_buffer_p.data())); // CPU pointer swaps
		}
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]-> enqueue_transfer_insert_field(lbm_domain[d]->kernel_transfer[field][1], 0u, bytes_per_cell); // PCIe copy + selective in-VRAM copy (x)
	}
	if(Dy>1u) { // communicate in y-direction
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_transfer_extract_field(lbm_domain[d]->kernel_transfer[field][0], 1u, bytes_per_cell); // selective in-VRAM copy (y) + PCIe copy
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
		for(uint d=0u; d<get_D(); d++) {
			const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy), dyp=x+(((y+1u)%Dy)+z*Dy)*Dx; // d = x+(y+z*Dy)*Dx
			lbm_domain[d]->transfer_buffer_p.exchange_host_buffer(lbm_domain[dyp]->transfer_buffer_m.exchange_host_buffer(lbm_domain[d]->transfer_buffer_p.data())); // CPU pointer swaps
		}
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]-> enqueue_transfer_insert_field(lbm_domain[d]->kernel_transfer[field][1], 1u, bytes_per_cell); // PCIe copy + selective in-VRAM copy (y)
	}
	if(Dz>1u) { // communicate in z-direction
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->enqueue_transfer_extract_field(lbm_domain[d]->kernel_transfer[field][0], 2u, bytes_per_cell); // selective in-VRAM copy (z) + PCIe copy
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
		for(uint d=0u; d<get_D(); d++) {
			const uint x=(d%(Dx*Dy))%Dx, y=(d%(Dx*Dy))/Dx, z=d/(Dx*Dy), dzp=x+(y+((z+1u)%Dz)*Dy)*Dx; // d = x+(y+z*Dy)*Dx
			lbm_domain[d]->transfer_buffer_p.exchange_host_buffer(lbm_domain[dzp]->transfer_buffer_m.exchange_host_buffer(lbm_domain[d]->transfer_buffer_p.data())); // CPU pointer swaps
		}
		for(uint d=0u; d<get_D(); d++) lbm_domain[d]-> enqueue_transfer_insert_field(lbm_domain[d]->kernel_transfer[field][1], 2u, bytes_per_cell); // PCIe copy + selective in-VRAM copy (z)
	}
}

void LBM::communicate_fi() {
	communicate_field(enum_transfer_field::fi, lbm_domain[0]->get_transfers()*sizeof(fpxx));
}
void LBM::communicate_rho_u_flags() {
	communicate_field(enum_transfer_field::rho_u_flags, 17u);
}
void LBM::communicate_flags() {
	communicate_field(enum_transfer_field::flags, 1u);
}
#ifdef FORCE_FIELD
void LBM::communicate_F() {
	communicate_field(enum_transfer_field::F, 12u);
}
#endif // FORCE_FIELD
#ifdef SURFACE
void LBM::communicate_phi_massex_flags() {
	communicate_field(enum_transfer_field::phi_massex_flags, 9u);
}
#endif // SURFACE
#ifdef TEMPERATURE
void LBM::communicate_gi() {
	communicate_field(enum_transfer_field::gi, sizeof(fpxx));
}
void LBM::communicate_T() {
	communicate_field(enum_transfer_field::T, 4u);
}
#endif // TEMPERATURE
#ifdef PARTICLES
void LBM::communicate_particles() {
	if(get_D()>1u) {
		if(initialized) {
			for(uint d=0u; d<get_D(); d++) lbm_domain[d]->particles.enqueue_read_from_device();
			for(uint d=0u; d<get_D(); d++) lbm_domain[d]->finish_queue(); // domain synchronization barrier
			for(ulong n=0ull; n<lbm_domain[0]->particles.length(); n++) { // parallel_for(lbm_domain[0]->particles.length(), [&](ulong n) {
				for(uint d=1u; d<get_D(); d++) { // gather modified particle positions
					const float lbm_domain_d___particles_x_n_ = lbm_domain[d]->particles.x[n];
					if(as_uint(lbm_domain_d___particles_x_n_)!=0xFFFFFFFFu) { // particle was in domain d and has been modified
						lbm_domain[0]->particles.x[n] = lbm_domain_d___particles_x_n_;
						lbm_domain[0]->particles.y[n] = lbm_domain[d]->particles.y[n];
						lbm_domain[0]->particles.z[n] = lbm_domain[d]->particles.z[n];
						break; // particle can only be in one domain at a time, no need to check other domains once it has been found
					}
				}
			} // });
		}
		for(uint d=0u; d<get_D(); d++) { // broadcast unified particle positions, using pointer of lbm_domain[0] instead of memory copy
			float* lbm_domain_d_particles_data = lbm_domain[d]->particles.exchange_host_buffer(lbm_domain[0]->particles.data());
			lbm_domain[d]->particles.enqueue_write_to_device();
			lbm_domain[d]->particles.exchange_host_buffer(lbm_domain_d_particles_data);
		}
	}
}
#endif // PARTICLES