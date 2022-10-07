
texture<float, cudaTextureType2D> windstress_X_current;
texture<float, cudaTextureType2D> windstress_X_next;

texture<float, cudaTextureType2D> windstress_Y_current;
texture<float, cudaTextureType2D> windstress_Y_next;

texture<float, cudaTextureType2D> atmospheric_pressure_current;
texture<float, cudaTextureType2D> atmospheric_pressure_next;


/**
  * Returns the wind stress, trilinearly interpolated in space and time
  * @param wind_stress_t_ \in [0, 1] determines the temporal interpolation (0=current, 1=next)
  * @param ti_ Location of this thread along the x-axis in number of cells (NOTE: half indices, including ghost cells)
  * @param tj_ Location of this thread along the y-axis in number of cells (NOTE: half indices, including ghost cells)
  * @param nx_ Number of cells along x axis including ghost cells
  * @param ny_ Number of cells along y axis including ghost cells
  */
__device__ float windStressX(float wind_stress_t_, float ti_, float tj_, int nx_, int ny_) {
    
    //Normalize coordinates (to [0, 1])
    const float s = ti_ / float(nx_);
    const float t = tj_ / float(ny_);
    
    //Look up current and next timestep (using bilinear texture interpolation)
    const float current = tex2D(windstress_X_current, s, t);
    const float next = tex2D(windstress_X_next, s, t);
    
    //Interpolate in time
    return wind_stress_t_*next + (1.0f - wind_stress_t_)*current;
}

/**
  * Returns the wind stress, trilinearly interpolated in space and time
  * @param wind_stress_t_ \in [0, 1] determines the temporal interpolation (0=current, 1=next)
  * @param ti_ Location of this thread along the x-axis in number of cells (NOTE: half indices)
  * @param tj_ Location of this thread along the y-axis in number of cells (NOTE: half indices)
  * @param nx_ Number of cells along x axis
  * @param ny_ Number of cells along y axis
  */
__device__ float windStressY(float wind_stress_t_, float ti_, float tj_, int nx_, int ny_) {
    
    //Normalize coordinates (to [0, 1])
    const float s = ti_ / float(nx_);
    const float t = tj_ / float(ny_);
    
    //Look up current and next timestep (using bilinear texture interpolation)
    const float current = tex2D(windstress_Y_current, s, t);
    const float next = tex2D(windstress_Y_next, s, t);
    
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
__device__ float2 atmospheric_pressure_central_diff(float atmospheric_pressure_t_, float ti_, float tj_, int nx_, int ny_) {
    
    //Normalize coordinates (to [0, 1])
    const float s = ti_ / float(nx_);
    const float t = tj_ / float(ny_);
    
    //Normalizing difference to neibouring coordinates
    const float dx = 1.0f/ float(nx_);
    const float dy = 1.0f/ float(ny_);
    
    //Look up current and next timestep (using bilinear texture interpolation)
    const float current_n = tex2D(atmospheric_pressure_current, s     , t + dy);
    const float next_n    = tex2D(atmospheric_pressure_next,    s     , t + dy);
    const float current_s = tex2D(atmospheric_pressure_current, s     , t - dy);
    const float next_s    = tex2D(atmospheric_pressure_next,    s     , t - dy);
    const float current_e = tex2D(atmospheric_pressure_current, s + dx, t     );
    const float next_e    = tex2D(atmospheric_pressure_next,    s + dx, t     );
    const float current_w = tex2D(atmospheric_pressure_current, s - dx, t     );
    const float next_w    = tex2D(atmospheric_pressure_next,    s - dx, t     );
    
    //Interpolate in time
    const float atm_p_n = atmospheric_pressure_t_*next_n + (1.0f - atmospheric_pressure_t_)*current_n;
    const float atm_p_s = atmospheric_pressure_t_*next_s + (1.0f - atmospheric_pressure_t_)*current_s;
    const float atm_p_e = atmospheric_pressure_t_*next_e + (1.0f - atmospheric_pressure_t_)*current_e;
    const float atm_p_w = atmospheric_pressure_t_*next_w + (1.0f - atmospheric_pressure_t_)*current_w;
    
    // Return central differences
    return make_float2(atm_p_e - atm_p_w, atm_p_n - atm_p_s);
}