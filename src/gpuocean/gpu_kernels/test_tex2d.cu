texture<float, cudaTextureType2D> data_tex;

extern "C" {

__global__ void textureTest(
        //Discretization parameters
        int nx_, int ny_,
        
        //Data
        float* data_ptr_, int data_pitch_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = bx + tx; 
    const int tj = by + ty;
    
    float* data_row = (float*) ((char*) data_ptr_ + data_pitch_*tj);
    
    if (ti < nx_ && tj < ny_) {
        float sx = (ti+0.5f)/float(nx_);
        float sy = (tj+0.5f)/float(ny_);
        
        float data = tex2D(data_tex, sx, sy);
        
        data_row[ti] = data;
    }
}

__device__ float bilinearInterpolate(float* data,  int data_nx, int data_ny, float norm_x, float norm_y) {
/**
 * Performs bilinear interpolation on a 2D grid of data points using normalized coordinates.
 *
 * Calculates an interpolated value within a data grid based on normalized (0 to 1) x and y coordinates.
 * Input coordinates outside [0,1] are clamped, with [0,0] mapping to data[0] and [1,1] to data[data_nx*data_ny - 1].
 *
 * Bilinear interpolation formula:
 * I = (1 - dx) * (1 - dy) * I00 + dx * (1 - dy) * I01 + (1 - dx) * dy * I10 + dx * dy * I11
 * Optimised formula to reduce multiplications:
 *  I = I00 + dx * (I01 - I00) + dy * (I10 - I00) + dx * dy * (I00 + I11 - I01 - I10)
 * where
 * - I00, I01, I10, and I11 are the values at the four surrounding grid points,
 * - dx and dy are the distances of the interpolated point from the grid points in the x and y directions, respectively.
 *
 * @param data Pointer to the 2D data array (row-major order).
 * @param data_nx Number of columns in the data array.
 * @param data_ny Number of rows in the data array.
 * @param norm_x Normalized x-coordinate of the interpolation point, \in [0, 1].
 * @param norm_y Normalized y-coordinate of the interpolation point, \in [0, 1].
 *
 * Returns:
 * - The interpolated value at (norm_x, norm_y).
 */

    // Clamp normalized coordinates to [0, 1]
    norm_x = __saturatef(norm_x);
    norm_y = __saturatef(norm_y);

    // Scale normalised coordinates up to the source dimensions
    float x = norm_x * (data_nx-1);
    float y = norm_y * (data_ny-1);
    
    // Calculate the base indices (the lower left corner)
    int x0 = floorf(x);
    int y0 = floorf(y);

    // Calculate the fractional part of the x and y coordinates
    float dx = x - x0;
    float dy = y - y0;

    // Fetch the values of the four neighbors ensuring they are clamping to edges of data array
    float d00 = data[y0*data_nx + x0];
    float d01 = data[y0*data_nx + min(x0+1, data_nx-1)];
    float d10 = data[min(y0+1, data_ny-1)*data_nx + x0];
    float d11 = data[min(y0+1, data_ny-1)*data_nx + min(x0+1, data_nx-1)];

    // original formula
    /*
    float result = (1.0f - dx)*(1.0f - dy)*d00 + dx*(1.0f - dy)*d01 + (1.0f - dx)*dy*d10 + dx*dy*d11;
    
    // optimised with fma
    // Precompute terms to simplify expressions and potentially increase FMA usage
    float one_minus_dx = 1.0f - dx;
    float one_minus_dy = 1.0f - dy;
    float result = fmaf(one_minus_dx * one_minus_dy, d00, 
                    fmaf(dx * one_minus_dy, d01, 
                    fmaf(one_minus_dx * dy, d10, 
                    dx * dy * d11)));

    */


    // Reduce number of multiplications by grouping dx,dy
    float result = d00 + dx*(d01-d00) + dy*(d10-d00) + dx*dy*(d00+d11-d01-d10);
/*    

    // optimised with fma
    // fmaf ( float  x, float  y, float  z ) 
    // Compute x*y + z as a single operation.

    float s1 = d01-d00;
    float s2 = d10-d00;
    float s3 = d00+d11-d01-d10;

    float result = fmaf(dx, s1, d00) + fmaf(dx*dy, s3, dy*s2);

    // sequential to get correct rounding
    float result = fmaf(dx, s1, d00);
    result = fmaf(dy, s2, result);
    result = fmaf(dx*dy, s3, result);
*/

    //float result = fmaf(dx, s1, fmaf(dy, s2, fmaf(dx*dy, s3, d00)));


    return result;
}

__global__ void interpTest(
        // Discretization parameters
        int nx_, int ny_,
        
        // Output data
        float* output_data_ptr_, int output_data_pitch_,
        // Tex output data
        float* tex_output_data_ptr_, int tex_output_data_pitch_,
        // Data array to sample from
        float* data_array_ptr_, int data_nx,  int data_ny) {
    // Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = bx + tx; 
    const int tj = by + ty;
    
    float* output_data_row = (float*) ((char*) output_data_ptr_ + output_data_pitch_*tj);
    float* tex_output_data_row = (float*) ((char*) tex_output_data_ptr_ + tex_output_data_pitch_*tj);
    //float* data_array_row = (float*) ((char*) data_array_ptr_ + data_array_pitch_*tj);
    
    if (ti < nx_ && tj < ny_) {
        // Note that the use of tex2D here and in GPU Ocean causes border issue when subsampling
        // See: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=texture#table-lookup
        float tex_sx = (ti+0.5f)/float(nx_);
        float tex_sy = (tj+0.5f)/float(ny_);
        float tex_data = tex2D(data_tex, tex_sx, tex_sy);
        tex_output_data_row[ti] = tex_data;

        float sx = ti/float(nx_-1);
        float sy = tj/float(ny_-1);
        float data = bilinearInterpolate(data_array_ptr_, data_nx, data_ny, sx, sy);
        output_data_row[ti] = data;

        //float data = data_array_row[ti];

    }
}

} // extern "C"
