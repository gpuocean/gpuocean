#ifndef INTERPOLATION_CU
#define INTERPOLATION_CU

/*
This software is part of GPU Ocean. 

Copyright (C) 2024 SINTEF Digital

These CUDA kernels implement interpolation functionality that is shared
between multiple numerical schemes for solving the shallow water
equations.

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

#include "common.cu"

/**
  *  Calculating the coefficient matrix for bicubic interpolation.
  *  Input matrix is evaluation of function values and derivatives in the corners
  *  of a unit square:
  *  f = [[ f00,  f01,  fy00,  fy01],
  *       [ f10,  f11,  fy10,  fy11],
  *       [fx00, fx01, fxy00, fxy01],
  *       [fx10, fx11, fxy10, fxy11] ]
  */
__device__ Matrix4x4_d bicubic_interpolation_coefficients(const Matrix4x4_d f) {
    Matrix4x4_d b;
    // [[ 1.0f,  0.0f,  0.0f,  0.0f)],
    //  [ 0.0f,  0.0f,  1.0f,  0.0f)],
    //  [-3.0f,  3.0f, -2.0f, -1.0f)],
    //  [ 2.0f, -2.0f,  1.0f,  1.0f)]]
    b.m_row0.x= 1.0f; b.m_row0.y= 0.0f; b.m_row0.z= 0.0f; b.m_row0.w= 0.0f;
    b.m_row1.x= 0.0f; b.m_row1.y= 0.0f; b.m_row1.z= 1.0f; b.m_row1.w= 0.0f;
    b.m_row2.x=-3.0f; b.m_row2.y= 3.0f; b.m_row2.z=-2.0f; b.m_row2.w=-1.0f;
    b.m_row3.x= 2.0f; b.m_row3.y=-2.0f; b.m_row3.z= 1.0f; b.m_row3.w= 1.0f;

    // Obtain fb = f * b^T, but store the result as its transpose:
    // fb[row i, col j]   = f[row i] dot b^T[col j]
    //                    = f[row i] dot b[row j]
    // fb^T[row i, col j] = f[row j] dot b[row i]
    Matrix4x4_d fb_transpose;
    /*
    Below is the loop unrolled version of the following loop:
    for (int i = 0; i < 4; i++) {
        fb_transpose.m_row[i] = make_float4(dotProduct(f.m_row0, b.m_row[i]),
                                            dotProduct(f.m_row1, b.m_row[i]),
                                            dotProduct(f.m_row2, b.m_row[i]),
                                            dotProduct(f.m_row3, b.m_row[i]));
    }*/
    fb_transpose.m_row0.x = dotProduct(f.m_row0, b.m_row0);
    fb_transpose.m_row0.y = dotProduct(f.m_row1, b.m_row0);
    fb_transpose.m_row0.z = dotProduct(f.m_row2, b.m_row0);
    fb_transpose.m_row0.w = dotProduct(f.m_row3, b.m_row0);

    fb_transpose.m_row1.x = dotProduct(f.m_row0, b.m_row1);
    fb_transpose.m_row1.y = dotProduct(f.m_row1, b.m_row1);
    fb_transpose.m_row1.z = dotProduct(f.m_row2, b.m_row1);
    fb_transpose.m_row1.w = dotProduct(f.m_row3, b.m_row1);

    fb_transpose.m_row2.x = dotProduct(f.m_row0, b.m_row2);
    fb_transpose.m_row2.y = dotProduct(f.m_row1, b.m_row2);
    fb_transpose.m_row2.z = dotProduct(f.m_row2, b.m_row2);
    fb_transpose.m_row2.w = dotProduct(f.m_row3, b.m_row2);

    fb_transpose.m_row3.x = dotProduct(f.m_row0, b.m_row3);
    fb_transpose.m_row3.y = dotProduct(f.m_row1, b.m_row3);
    fb_transpose.m_row3.z = dotProduct(f.m_row2, b.m_row3);
    fb_transpose.m_row3.w = dotProduct(f.m_row3, b.m_row3);

    // Obtain out = b * f * b^T = b * fb
    // out[row i, col j] = b[row i] dot fb[col j]
    //                   = b[row i] dot fb^T[row j]
    Matrix4x4_d out;
    /*
    Below is the loop unrolled version of the following loop:
    for (int i = 0; i < 4; i++) {
        out.m_row[i] = make_float4(dotProduct(b.m_row[i], fb_transpose.m_row[0]),
                                   dotProduct(b.m_row[i], fb_transpose.m_row[1]),
                                   dotProduct(b.m_row[i], fb_transpose.m_row[2]),
                                   dotProduct(b.m_row[i], fb_transpose.m_row[3]));
    }*/

    out.m_row0.x = dotProduct(b.m_row0, fb_transpose.m_row0);
    out.m_row0.y = dotProduct(b.m_row0, fb_transpose.m_row1);
    out.m_row0.z = dotProduct(b.m_row0, fb_transpose.m_row2);
    out.m_row0.w = dotProduct(b.m_row0, fb_transpose.m_row3);

    out.m_row1.x = dotProduct(b.m_row1, fb_transpose.m_row0);
    out.m_row1.y = dotProduct(b.m_row1, fb_transpose.m_row1);
    out.m_row1.z = dotProduct(b.m_row1, fb_transpose.m_row2);
    out.m_row1.w = dotProduct(b.m_row1, fb_transpose.m_row3);

    out.m_row2.x = dotProduct(b.m_row2, fb_transpose.m_row0);
    out.m_row2.y = dotProduct(b.m_row2, fb_transpose.m_row1);
    out.m_row2.z = dotProduct(b.m_row2, fb_transpose.m_row2);
    out.m_row2.w = dotProduct(b.m_row2, fb_transpose.m_row3);

    out.m_row3.x = dotProduct(b.m_row3, fb_transpose.m_row0);
    out.m_row3.y = dotProduct(b.m_row3, fb_transpose.m_row1);
    out.m_row3.z = dotProduct(b.m_row3, fb_transpose.m_row2);
    out.m_row3.w = dotProduct(b.m_row3, fb_transpose.m_row3);

    return out;
}

__device__ float bicubic_evaluation(const float4 x, 
                                    const float4 y, 
                                    const Matrix4x4_d coeff) 
{
    // out = x^T * coeff * y
    // out = x^T * temp

    // tmp[i] = coeff[row i] * y
    float4 tmp;
    tmp.x = dotProduct(coeff.m_row0, y);
    tmp.y = dotProduct(coeff.m_row1, y);
    tmp.z = dotProduct(coeff.m_row2, y);
    tmp.w = dotProduct(coeff.m_row3, y);

    return dotProduct(x, tmp);
}


__device__ float bilinear_interpolation(const float* data,  int data_nx, int data_ny, float norm_x, float norm_y) {
/**
 * Performs bilinear interpolation on a 2D grid of data points using normalized coordinates.
 *
 * Calculates an interpolated value within a data grid based on normalized (0 to 1) x and y coordinates.
 * Matches tex2D normalized input coordinates.
 * Input coordinates outside [0,1] are clamped.
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

    norm_x -= 0.5f / data_nx;
    norm_y -= 0.5f / data_ny;

    norm_x = __saturatef(norm_x);
    norm_y = __saturatef(norm_y);

    // Scale normalised coordinates up to the source dimensions
    const float x = norm_x*data_nx;
    const float y = norm_y*data_ny;
    // Calculate the base indices (the lower left corner)
    const int x0 = floorf(x);
    const int y0 = floorf(y);

    // Calculate the fractional part of the x and y coordinates
    const float dx = x - x0;
    const float dy = y - y0;

    // Fetch the values of the four neighbors ensuring they are clamping to edges of data array
    const float d00 = data[y0*data_nx + x0];
    const float d01 = data[y0*data_nx + min(x0+1, data_nx-1)];
    const float d10 = data[min(y0+1, data_ny-1)*data_nx + x0];
    const float d11 = data[min(y0+1, data_ny-1)*data_nx + min(x0+1, data_nx-1)];

    // Reduce number of multiplications by grouping dx,dy
    const float result = d00 + dx*(d01-d00) + dy*(d10-d00) + dx*dy*(d00+d11-d01-d10);

    return result;
}


#endif // INTERPOLATION_CU
