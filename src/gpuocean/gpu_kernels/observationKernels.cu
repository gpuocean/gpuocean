/*
This software is part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This CUDA kernel implements observation operators.

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



/**
  * Kernel that observes the velocity of the ocean flow in the drifter positions
  */
extern "C" {
__global__ void observeUnderlyingFlow(
        //Discretization parameters
        int nx_, int ny_,
        float dx_, float dy_,

        int x_zero_reference_cell_, // the cell column representing x0 (x0 at western face)
        int y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)
	
    	// Data
        float* eta_ptr_, int eta_pitch_,
        float* hu_ptr_, int hu_pitch_,
        float* hv_ptr_, int hv_pitch_,
	    // H should be read from buffer, but for now we use a constant value
        //__global float* H_ptr_, int H_pitch_,
        float H_,


        int num_drifters_,
        float* drifters_positions_, int drifters_pitch_,
        float* observation_ptr_, int observation_pitch_
    ) {

    //Index of thread within block (only needed in one dim)
    const int tx = threadIdx.x;
        
    //Index of block within domain (only needed in one dim)
    const int bx = blockDim.x * blockIdx.x;
        
    //Index of drifter that decides the position we will observe
    const int ti = bx + tx;
    
    if (ti < num_drifters_) {
	// Obtain pointer to my drifter:
	float* drifter = (float*) ((char*) drifters_positions_ + drifters_pitch_*ti);
	float drifter_pos_x = drifter[0];
	float drifter_pos_y = drifter[1];

	// Find cell ID for the cell in which our particle is
	int const cell_id_x = (int)(floor(drifter_pos_x/dx_) + x_zero_reference_cell_);
	int const cell_id_y = (int)(floor(drifter_pos_y/dy_) + y_zero_reference_cell_);

	float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*cell_id_y);
	float const hu = hu_row[cell_id_x];

	float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*cell_id_y);
	float const hv = hv_row[cell_id_x];

    // Obtain pointer to my observation:
    float* observation = (float*)((char*) observation_ptr_ + observation_pitch_*ti);
    observation[0] = hu;
    observation[1] = hv;
    }
}
} // extern "C"

