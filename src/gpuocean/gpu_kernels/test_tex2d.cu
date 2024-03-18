#include "common.cu"

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

__global__ void interpTest(
        // Discretization parameters
        int nx_, int ny_,
        
        // Output data
        float* output_data_ptr_, int output_data_pitch_,
        // Tex output data
        float* tex_output_data_ptr_, int tex_output_data_pitch_,
        // Data array to sample from
        const float* data_array_ptr_, int data_nx,  int data_ny) {
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
        float data = bilinear_interpolation(data_array_ptr_, data_nx, data_ny, sx, sy);
        output_data_row[ti] = data;

        //float data = data_array_row[ti];

    }
}

} // extern "C"
