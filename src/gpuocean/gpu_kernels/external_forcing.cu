
texture<float, cudaTextureType2D> windstress_X_current;
texture<float, cudaTextureType2D> windstress_X_next;

texture<float, cudaTextureType2D> windstress_Y_current;
texture<float, cudaTextureType2D> windstress_Y_next;


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
    float current = tex2D(windstress_X_current, s, t);
    float next = tex2D(windstress_X_next, s, t);
    
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
    float current = tex2D(windstress_Y_current, s, t);
    float next = tex2D(windstress_Y_next, s, t);
    
    //Interpolate in time
    return wind_stress_t_*next + (1.0f - wind_stress_t_)*current;
}
