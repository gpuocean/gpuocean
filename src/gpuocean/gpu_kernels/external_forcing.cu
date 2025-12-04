
#include "interpolation.cu"


/**
  * Returns the wind stress, trilinearly interpolated in space and time
  * @param wind_stress_current_arr Array of current windstress values to interpolate from
  * @param wind_stress_next_arr Array of next windstress values to interpolate from
  * @param wind_stress_t_ \in [0, 1] determines the temporal interpolation (0=current, 1=next)
  * @param ti_ Location of this thread along the x-axis in number of cells (NOTE: half indices, including ghost cells)
  * @param tj_ Location of this thread along the y-axis in number of cells (NOTE: half indices, including ghost cells)
  * @param nx_ Number of cells along x axis including ghost cells
  * @param ny_ Number of cells along y axis including ghost cells
  * @param data_nx Number of cells along x axis for the windstress arrays
  * @param data_ny Number of cells along y axis for the windstress arrays
  */
__device__ float windStress(const float* wind_stress_current_arr, const float* wind_stress_next_arr, float wind_stress_t_, float ti_, float tj_, int nx_, int ny_, int data_nx, int data_ny) {
    
    //Normalize coordinates (to [0, 1])
    const float s = ti_ / float(nx_);
    const float t = tj_ / float(ny_);
    
    //Look up current and next timestep (using bilinear texture interpolation)
    const float current = bilinear_interpolation(wind_stress_current_arr, data_nx, data_ny, s, t);
    const float next = bilinear_interpolation(wind_stress_next_arr, data_nx, data_ny, s, t);
    
    //Interpolate in time
    return wind_stress_t_*next + (1.0f - wind_stress_t_)*current;
}


/**
  * Returns the spatial central differences of the atmospheric pressure, based on trilinearly interpolated in space and time
  * @param atmospheric_pressure_t_ \in [0, 1] determines the temporal interpolation (0=current, 1=next)
  * @param ti_ Location of this thread along the x-axis in number of cells (NOTE: half indices, including ghost cells)
  * @param tj_ Location of this thread along the y-axis in number of cells (NOTE: half indices, including ghost cells)
  * @param nx_ Number of cells along x axis including ghost cells
  * @param ny_ Number of cells along y axis including ghost cells
  */
__device__ float2 atmospheric_pressure_central_diff(const float* atmospheric_pressure_current_arr, const float* atmospheric_pressure_next_arr, float atmospheric_pressure_t_, float ti_, float tj_, int nx_, int ny_, int data_nx, int data_ny) {

    //Normalize coordinates (to [0, 1])
    const float s = ti_ / float(nx_);
    const float t = tj_ / float(ny_);

    //Normalizing difference to neibouring coordinates
    const float dx = 1.0f/ float(nx_);
    const float dy = 1.0f/ float(ny_);

    //Look up current and next timestep (using bilinear texture interpolation)
    const float current_n = bilinear_interpolation(atmospheric_pressure_current_arr, data_nx, data_ny, s,      t + dy);
    const float next_n    = bilinear_interpolation(atmospheric_pressure_next_arr,    data_nx, data_ny, s,      t + dy);
    const float current_s = bilinear_interpolation(atmospheric_pressure_current_arr, data_nx, data_ny, s,      t - dy);
    const float next_s    = bilinear_interpolation(atmospheric_pressure_next_arr,    data_nx, data_ny, s,      t - dy);
    const float current_e = bilinear_interpolation(atmospheric_pressure_current_arr, data_nx, data_ny, s + dx, t     );
    const float next_e    = bilinear_interpolation(atmospheric_pressure_next_arr,    data_nx, data_ny, s + dx, t     );
    const float current_w = bilinear_interpolation(atmospheric_pressure_current_arr, data_nx, data_ny, s - dx, t     );
    const float next_w    = bilinear_interpolation(atmospheric_pressure_next_arr,    data_nx, data_ny, s - dx, t     );

    //Interpolate in time
    const float atm_p_n = atmospheric_pressure_t_*next_n + (1.0f - atmospheric_pressure_t_)*current_n;
    const float atm_p_s = atmospheric_pressure_t_*next_s + (1.0f - atmospheric_pressure_t_)*current_s;
    const float atm_p_e = atmospheric_pressure_t_*next_e + (1.0f - atmospheric_pressure_t_)*current_e;
    const float atm_p_w = atmospheric_pressure_t_*next_w + (1.0f - atmospheric_pressure_t_)*current_w;

    // Return central differences
    return make_float2(atm_p_e - atm_p_w, atm_p_n - atm_p_s);
}

/**
  * Returns the spatial central differences of the atmospheric pressure for same sized grids, based on interpolation in time and lookup in space.
  * @param atmospheric_pressure_t_ \in [0, 1] determines the temporal interpolation (0=current, 1=next)
  * @param ti Location of this thread along the x-axis in number of cells (NOTE: half indices, including ghost cells)
  * @param tj Location of this thread along the y-axis in number of cells (NOTE: half indices, including ghost cells)
  */
__device__ float2 atmospheric_pressure_central_diff_lookup(const float* atmospheric_pressure_current_arr, const float* atmospheric_pressure_next_arr, float atmospheric_pressure_t_, int ti, int tj, int data_nx) {


    //Look up current and next timestep (using bilinear texture interpolation)
    const float current_n = atmospheric_pressure_current_arr[(tj+1)*data_nx + (ti+1)];
    const float next_n    = atmospheric_pressure_next_arr   [(tj+1)*data_nx + (ti+1)];
    const float current_s = atmospheric_pressure_current_arr[(tj-1)*data_nx + (ti+1)];
    const float next_s    = atmospheric_pressure_next_arr   [(tj-1)*data_nx + (ti+1)];
    const float current_e = atmospheric_pressure_current_arr[(tj  )*data_nx + (ti+1)];
    const float next_e    = atmospheric_pressure_next_arr   [(tj  )*data_nx + (ti+1)];
    const float current_w = atmospheric_pressure_current_arr[(tj  )*data_nx + (ti-1)];
    const float next_w    = atmospheric_pressure_next_arr   [(tj  )*data_nx + (ti-1)];

    //Interpolate in time
    const float atm_p_n = atmospheric_pressure_t_*next_n + (1.0f - atmospheric_pressure_t_)*current_n;
    const float atm_p_s = atmospheric_pressure_t_*next_s + (1.0f - atmospheric_pressure_t_)*current_s;
    const float atm_p_e = atmospheric_pressure_t_*next_e + (1.0f - atmospheric_pressure_t_)*current_e;
    const float atm_p_w = atmospheric_pressure_t_*next_w + (1.0f - atmospheric_pressure_t_)*current_w;

    // Return central differences
    return make_float2(atm_p_e - atm_p_w, atm_p_n - atm_p_s);
}