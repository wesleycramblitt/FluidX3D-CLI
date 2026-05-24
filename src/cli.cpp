#include "fluidx3d.h"
#include "lbm.hpp"
#include "info.hpp"
#include "setup.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

void main_physics();

static void print_usage() {
	printf("Usage: fluidx3d [OPTIONS] COMMAND [ARGS]\n\n");
	printf("Commands:\n");
	printf("  run      Run simulation to completion\n");
	printf("  step     Run N steps, optionally export field data\n");
	printf("  probe    Read field values at specific grid coordinates\n");
	printf("  info     Print device info and exit\n");
	printf("  config   Print current configuration\n\n");
	printf("Grid & Physics:\n");
	printf("  --nx, --ny, --nz N      Grid resolution (default 256,256,256)\n");
	printf("  --nu FLOAT              Kinematic viscosity\n");
	printf("  --fx, --fy, --fz FLOAT  Volume force\n");
	printf("  --sigma FLOAT           Surface tension\n");
	printf("  --steps N               Total time steps\n\n");
	printf("LBM Variant:\n");
	printf("  --velocity-set N        9,15,19,27 (default 19)\n");
	printf("  --collision SRT|TRT     (default SRT)\n");
	printf("  --extensions LIST       Comma-separated: volume-force,force-field,\n");
	printf("                           equilibrium, moving, surface, temperature,\n");
	printf("                           subgrid, particles\n\n");
	printf("STL Mesh:\n");
	printf("  --stl PATH              Binary .stl file to voxelize\n");
	printf("  --stl-scale FLOAT       Mesh size (0=auto-fill box, >0=exact length)\n");
	printf("  --stl-center X,Y,Z      Mesh center in grid coords\n");
	printf("  --stl-rotation ANG,AX,AY,AZ  Rotation angle° + axis\n");
	printf("  --stl-flag N            Cell flag (default 1 = TYPE_S)\n\n");
	printf("Output:\n");
	printf("  --export-path DIR       Output directory for VTK\n");
	printf("  --field NAME            Field to export/probe (rho, u, flags, force, phi, T, particles)\n");
	printf("  --probe X,Y,Z           Grid coordinates to probe (with 'probe' command)\n\n");
	printf("Other:\n");
	printf("  --devices LIST          GPU device IDs (comma-separated)\n");
	printf("  --help, -h              Show this help\n\n");
	printf("Examples:\n");
	printf("  fluidx3d step --nx 64 --ny 64 --nz 64 --nu 0.02 --steps 500\n");
	printf("  fluidx3d step --nx 128 --ny 256 --nz 128 --nu 0.01 --stl cow.stl --stl-scale 0 \\\n");
	printf("               --extensions surface,volume-force --sigma 0.001 --steps 200\n");
	printf("  fluidx3d probe --nx 64 --ny 64 --nz 64 --nu 0.02 --steps 100 \\\n");
	printf("               --probe 32,32,32 --field u\n");
	printf("  fluidx3d info\n");
}

static void print_devices() {
	const vector<Device_Info>& devices = get_devices();
	printf("OpenCL devices found: %zu\n\n", devices.size());
	for(uint i=0u; i<(uint)devices.size(); i++) {
		const Device_Info& d = devices[i];
		printf("Device %u: %s\n", i, d.name.c_str());
		printf("  Vendor:    %s\n", d.vendor.c_str());
		printf("  Driver:    %s\n", d.driver_version.c_str());
		printf("  OpenCL:    %s\n", d.opencl_c_version.c_str());
		printf("  Compute:   %u units @ %u MHz\n", d.compute_units, d.clock_frequency);
		printf("  VRAM:      %u MB\n", d.memory);
		printf("  TFLOPs:    %.3f\n\n", d.tflops);
	}
}

static int field_id_from_name(const char* name) {
	if(!name) return -1;
	if(strcmp(name, "rho") == 0)  return FLUIDX3D_FIELD_RHO;
	if(strcmp(name, "u") == 0)    return FLUIDX3D_FIELD_U;
	if(strcmp(name, "flags") == 0) return FLUIDX3D_FIELD_FLAGS;
	if(strcmp(name, "force") == 0) return FLUIDX3D_FIELD_FORCE;
	if(strcmp(name, "phi") == 0)  return FLUIDX3D_FIELD_PHI;
	if(strcmp(name, "T") == 0 || strcmp(name, "temp") == 0) return FLUIDX3D_FIELD_TEMP;
	if(strcmp(name, "particles") == 0) return FLUIDX3D_FIELD_PARTICLES;
	return -1;
}

static void print_field_stats(FluidX3D_Solver* solver, int field_id) {
	uint64_t count;
	float* data = fluidx3d_read_field_float(solver, field_id, &count);
	if(!data || count == 0) {
		printf("  (field not available)\n");
		return;
	}
	// Compute min, max, mean
	float vmin = data[0], vmax = data[0];
	double sum = 0.0;
	for(uint64_t i=0; i<count; i++) {
		if(data[i] < vmin) vmin = data[i];
		if(data[i] > vmax) vmax = data[i];
		sum += data[i];
	}
	double mean = sum / (double)count;
	printf("  count=%llu  min=%.6f  max=%.6f  mean=%.6f\n",
	       (unsigned long long)count, vmin, vmax, mean);
	free(data);
}

static void parse_float3(const char* s, float* x, float* y, float* z) {
	char buf[256]; strncpy(buf, s, 255); buf[255]=0;
	char* t = strtok(buf, ",");
	if(t) *x = (float)atof(t);
	t = strtok(nullptr, ",");
	if(t) *y = (float)atof(t);
	t = strtok(nullptr, ",");
	if(t) *z = (float)atof(t);
}

int main(int argc, char** argv) {
	if(argc < 2) {
		// Legacy mode: run default setup (2D Taylor-Green)
		info.allow_printing.lock();
		main_arguments = get_main_arguments(argc, argv);
		thread compute_thread(main_physics);
		info.allow_printing.unlock();
		do {
			info.print_update();
			sleep(0.050);
		} while(running);
		compute_thread.join();
		return 0;
	}

	const char* cmd = argv[1];

	if(strcmp(cmd, "info") == 0) { print_devices(); return 0; }
	if(strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) { print_usage(); return 0; }

	// ── Parse options ──
	FluidX3D_Config config = fluidx3d_default_config();
	uint64_t steps = UINT64_MAX;
	const char* export_path = nullptr;
	const char* stl_path = nullptr;
	float stl_scale = 0.0f;
	float stl_center_x=0, stl_center_y=0, stl_center_z=0;
	float stl_rot_angle=0, stl_rot_ax=0, stl_rot_ay=0, stl_rot_az=1;
	int stl_flag = 1;
	const char* field_name = nullptr;
	float probe_x=0, probe_y=0, probe_z=0;
	bool has_probe = false;

	for(int i=2; i<argc; i++) {
		if(strcmp(argv[i], "--nx") == 0 && i+1<argc) config.nx = (uint32_t)atoi(argv[++i]);
		else if(strcmp(argv[i], "--ny") == 0 && i+1<argc) config.ny = (uint32_t)atoi(argv[++i]);
		else if(strcmp(argv[i], "--nz") == 0 && i+1<argc) config.nz = (uint32_t)atoi(argv[++i]);
		else if(strcmp(argv[i], "--nu") == 0 && i+1<argc) config.nu = (float)atof(argv[++i]);
		else if(strcmp(argv[i], "--fx") == 0 && i+1<argc) config.fx = (float)atof(argv[++i]);
		else if(strcmp(argv[i], "--fy") == 0 && i+1<argc) config.fy = (float)atof(argv[++i]);
		else if(strcmp(argv[i], "--fz") == 0 && i+1<argc) config.fz = (float)atof(argv[++i]);
		else if(strcmp(argv[i], "--sigma") == 0 && i+1<argc) config.sigma = (float)atof(argv[++i]);
		else if(strcmp(argv[i], "--steps") == 0 && i+1<argc) steps = (uint64_t)atoll(argv[++i]);
		else if(strcmp(argv[i], "--velocity-set") == 0 && i+1<argc) config.velocity_set = atoi(argv[++i]);
		else if(strcmp(argv[i], "--collision") == 0 && i+1<argc)
			config.collision = strcmp(argv[++i], "TRT") == 0 ? 1 : 0;
		else if(strcmp(argv[i], "--extensions") == 0 && i+1<argc) {
			config.extensions = 0;
			char buf[256]; strncpy(buf, argv[++i], 255); buf[255]=0;
			char* tok = strtok(buf, ",");
			while(tok) {
				while(*tok==' ') tok++;
				if(strcmp(tok, "volume-force") == 0) config.extensions |= FLUIDX3D_EXT_VOLUME_FORCE;
				else if(strcmp(tok, "force-field") == 0) config.extensions |= FLUIDX3D_EXT_FORCE_FIELD;
				else if(strcmp(tok, "equilibrium") == 0) config.extensions |= FLUIDX3D_EXT_EQUILIBRIUM_BOUNDARIES;
				else if(strcmp(tok, "moving") == 0) config.extensions |= FLUIDX3D_EXT_MOVING_BOUNDARIES;
				else if(strcmp(tok, "surface") == 0) config.extensions |= FLUIDX3D_EXT_SURFACE;
				else if(strcmp(tok, "temperature") == 0) config.extensions |= FLUIDX3D_EXT_TEMPERATURE;
				else if(strcmp(tok, "subgrid") == 0) config.extensions |= FLUIDX3D_EXT_SUBGRID;
				else if(strcmp(tok, "particles") == 0) config.extensions |= FLUIDX3D_EXT_PARTICLES;
				tok = strtok(nullptr, ",");
			}
		}
		else if(strcmp(argv[i], "--stl") == 0 && i+1<argc) stl_path = argv[++i];
		else if(strcmp(argv[i], "--stl-scale") == 0 && i+1<argc) stl_scale = (float)atof(argv[++i]);
		else if(strcmp(argv[i], "--stl-center") == 0 && i+1<argc) parse_float3(argv[++i], &stl_center_x, &stl_center_y, &stl_center_z);
		else if(strcmp(argv[i], "--stl-rotation") == 0 && i+1<argc) { parse_float3(argv[++i], &stl_rot_angle, &stl_rot_ax, &stl_rot_ay); }
		else if(strcmp(argv[i], "--stl-flag") == 0 && i+1<argc) stl_flag = atoi(argv[++i]);
		else if(strcmp(argv[i], "--field") == 0 && i+1<argc) field_name = argv[++i];
		else if(strcmp(argv[i], "--probe") == 0 && i+1<argc) { parse_float3(argv[++i], &probe_x, &probe_y, &probe_z); has_probe = true; }
		else if(strcmp(argv[i], "--devices") == 0 && i+1<argc) {
			const char* devs = argv[++i];
			config.device_count = 0;
			char buf[256]; strncpy(buf, devs, 255); buf[255]=0;
			char* tok = strtok(buf, ",");
			while(tok && config.device_count < 8) {
				config.device_ids[config.device_count++] = atoi(tok);
				tok = strtok(nullptr, ",");
			}
		}
		else if(strcmp(argv[i], "--export-path") == 0 && i+1<argc) export_path = argv[++i];
		else if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { print_usage(); return 0; }
	}

	// ── config command ──
	if(strcmp(cmd, "config") == 0) {
		printf("Grid:          %ux%ux%u\n", config.nx, config.ny, config.nz);
		printf("Viscosity:     %f\n", config.nu);
		printf("Force:         %f, %f, %f\n", config.fx, config.fy, config.fz);
		printf("Velocity set:  D%uQ%u\n", config.velocity_set==9?2:3, config.velocity_set);
		printf("Collision:     %s\n", config.collision?"TRT":"SRT");
		printf("Extensions:    0x%X\n", config.extensions);
		if(stl_path) printf("STL:           %s (scale=%.2f, flag=%d)\n", stl_path, stl_scale, stl_flag);
		return 0;
	}

	// ── run / step / probe ──
	bool is_run   = (strcmp(cmd, "run") == 0);
	bool is_step  = (strcmp(cmd, "step") == 0);
	bool is_probe = (strcmp(cmd, "probe") == 0);

	if(!is_run && !is_step && !is_probe) { print_usage(); return 1; }

	if(steps == UINT64_MAX && is_step) steps = 1;

	info.print_logo();
	config.total_steps = steps;

	FluidX3D_Solver* solver = fluidx3d_create(&config);
	if(!solver) { fprintf(stderr, "Error: Failed to create solver\n"); return 1; }

	// Voxelize STL if provided
	if(stl_path) {
		// Auto-center: if center not specified, use grid center
		if(stl_center_x==0 && stl_center_y==0 && stl_center_z==0) {
			stl_center_x = (float)config.nx * 0.5f;
			stl_center_y = (float)config.ny * 0.5f;
			stl_center_z = (float)config.nz * 0.5f;
		}
		printf("Voxelizing STL: %s (scale=%.2f, center=%.0f,%.0f,%.0f)\n",
		       stl_path, stl_scale, stl_center_x, stl_center_y, stl_center_z);
		fluidx3d_voxelize_stl(solver, stl_path,
			stl_center_x, stl_center_y, stl_center_z,
			stl_rot_angle, stl_rot_ax, stl_rot_ay, stl_rot_az,
			stl_scale, (uint8_t)stl_flag);
	}

	fluidx3d_initialize(solver);
	fluidx3d_run(solver, steps, steps);

	uint32_t nx, ny, nz;
	fluidx3d_get_dims(solver, &nx, &ny, &nz);
	printf("\nCompleted %llu steps. Grid: %ux%ux%u\n",
	       (unsigned long long)fluidx3d_get_step(solver), nx, ny, nz);

	// ── Field output ──
	int fid = field_id_from_name(field_name);
	if(!field_name && !has_probe) fid = FLUIDX3D_FIELD_U; // default: show velocity stats

	if(fid >= 0) {
		printf("Field '%s': ", field_name ? field_name : "u");
		print_field_stats(solver, fid);
	}

	// ── Probe specific point ──
	if(has_probe) {
		uint32_t px = (uint32_t)probe_x, py = (uint32_t)probe_y, pz = (uint32_t)probe_z;
		if(px < nx && py < ny && pz < nz) {
			uint64_t N = (uint64_t)nx * (uint64_t)ny * (uint64_t)nz;
			uint64_t idx = (uint64_t)px + ((uint64_t)py + (uint64_t)pz * (uint64_t)ny) * (uint64_t)nx;

			// Read velocity
			uint64_t ucount;
			float* u = fluidx3d_read_field_float(solver, FLUIDX3D_FIELD_U, &ucount);
			if(u && ucount >= 3*N) {
				float vx = u[idx], vy = u[idx+N], vz = u[idx+2*N];
				float speed = sqrtf(vx*vx+vy*vy+vz*vz);
				printf("Probe (%u,%u,%u): velocity = (%f, %f, %f)  speed = %f\n",
				       px, py, pz, vx, vy, vz, speed);
				free(u);
			}

			// Read flags
			uint64_t fcount;
			uint8_t* flags = fluidx3d_read_field_uchar(solver, FLUIDX3D_FIELD_FLAGS, &fcount);
			if(flags && fcount >= N) {
				printf("               flags   = 0x%02X", flags[idx]);
				if(flags[idx] & 0x01) printf(" TYPE_S");
				if(flags[idx] & 0x02) printf(" TYPE_E");
				if(flags[idx] & 0x04) printf(" TYPE_T");
				if(flags[idx] & 0x08) printf(" TYPE_F");
				if(flags[idx] & 0x10) printf(" TYPE_I");
				if(flags[idx] & 0x20) printf(" TYPE_G");
				if(flags[idx] == 0) printf(" (fluid)");
				printf("\n");
				free(flags);
			}

			// Read density
			float* rho = fluidx3d_read_field_float(solver, FLUIDX3D_FIELD_RHO, &ucount);
			if(rho && ucount >= N) {
				printf("               density = %f\n", rho[idx]);
				free(rho);
			}
		} else {
			printf("Probe (%u,%u,%u): out of bounds (grid is %ux%ux%u)\n", px, py, pz, nx, ny, nz);
		}
	}

	// ── VTK export ──
	if(export_path) {
		if(!fid) fid = FLUIDX3D_FIELD_U;
		fluidx3d_export_vtk(solver, fid, export_path);
	}

	fluidx3d_destroy(solver);
	return 0;
}
