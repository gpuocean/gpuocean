/*
This software is part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

Implements interpolation using textures. 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

texture<float, cudaTextureType2D> my_texture_current;
texture<float, cudaTextureType2D> my_texture_next;

extern "C" {
__global__ void interpolationTest(
        //Discretization parameters
        int nx_, int ny_,
        float dx_, float dy_,
        float t_,
        
        //Data
        float* H_ptr_, int H_pitch_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = bx + tx; 
    const int tj = by + ty;
    
    float* H_row = (float*) ((char*) H_ptr_ + H_pitch_*tj);
    
    if (ti < nx_ && tj < ny_) {
        float sx = ti/float(nx_);
        float sy = tj/float(ny_);
        float current = tex2D(my_texture_current, sx, sy);
        float next = tex2D(my_texture_next, sx, sy);
        
        H_row[ti] = (1.0f-t_)*current + t_*next;
    }
}

} // extern "C"