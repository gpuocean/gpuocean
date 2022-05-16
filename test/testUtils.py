# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2017, 2018 SINTEF Digital

This python module provides common functionality shared between all
tests.

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
"""

import os
import sys
import numpy as np


testdir = 'timestep50'

def utils(a):
    return a+1

#import pyopencl
def make_cl_ctx():
    #Make sure we get compiler output from OpenCL
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

    #Set which CL device to use, and disable kernel caching
    if (str.lower(sys.platform).startswith("linux")):
        os.environ["PYOPENCL_CTX"] = "0"
    else:
        os.environ["PYOPENCL_CTX"] = "1"
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    os.environ["PYOPENCL_NO_CACHE"] = "1"
        
    #Create OpenCL context
    cl_ctx = pyopencl.create_some_context()
    #print "Using ", cl_ctx.devices[0].name
    return cl_ctx


## A common initial condition maker:
def makeCornerBump(eta, nx, ny, dx, dy, halo):
    x_center = 4*dx
    y_center = 4*dy
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] = np.exp(-(x**2/size+y**2/size))

def makeCentralBump(eta, nx, ny, dx, dy, halo):
    x_center = dx*nx/2.0
    y_center = dy*ny/2.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] = np.exp(-(x**2/size+y**2/size))

def makeUpperCornerBump(eta, nx, ny, dx, dy, halo):
    x_center = (nx-4)*dx
    y_center = (ny-4)*dy
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] = np.exp(-(x**2/size+y**2/size))
                
def makeLowerLeftBump(eta, nx, ny, dx, dy, halo):
    x_center = dx*nx*0.3
    y_center = dy*ny*0.2
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] = np.exp(-(x**2/size+y**2/size))

def makeBottomTopography(Hi, nx, ny, dx, dy, halo, intersections=True):
    extra_cells = 0
    if intersections:
        extra_cells = 1
    for j in range(-halo[2], ny + halo[0] + extra_cells):
        for i in range(-halo[3], nx + halo[1] + extra_cells):
            Hi[j+halo[2], i+halo[3]] = 6 + 2.0*np.cos(0.3*(i + i/(np.sin(0.5*j)+2.5))) + \
                                       2.0*np.sin(2*np.pi*(j+i)/(2.0*ny))
            

                
## Add initial conditions on top of existing ones:
def addCornerBump(eta, nx, ny, dx, dy, halo):
    x_center = 4*dx
    y_center = 4*dy
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] += np.exp(-(x**2/size+y**2/size))

def addCentralBump(eta, nx, ny, dx, dy, halo):
    x_center = dx*nx/2.0
    y_center = dy*ny/2.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] += np.exp(-(x**2/size+y**2/size))

def addUpperCornerBump(eta, nx, ny, dx, dy, halo):
    x_center = (nx-4)*dx
    y_center = (ny-4)*dy
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] += np.exp(-(x**2/size+y**2/size))

def makeBathymetryCrater(B, nx, ny, dx, dy, halo):
    x_center = dx*nx/2.0
    y_center = dy*ny/2.0
    minReach = min(nx*dx, ny*dy)
    innerEdge = minReach*0.3/2.0
    outerEdge = minReach*0.7/2.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            radius = np.sqrt(x**2 + y**2)
            if (radius > innerEdge) and (radius < outerEdge):
                B[j+halo[2], i+halo[3]] = 30.0*np.sin((radius - innerEdge)/(outerEdge - innerEdge)*np.pi )**2
            else:
                B[j+halo[2], i+halo[3]] = 0.0
                
def makeBathymetryCrazyness(B, nx, ny, dx, dy, halo):
    length = dx*nx*1.0
    height = dy*ny*1.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i*1.0
            y = dy*j*1.0
            B[j+halo[2], i+halo[3]] = 25.0*(np.sin(np.pi*(x/length)*4)**2 + np.sin(np.pi*(y/height)*4)**2)            

def saveResults(eta, u, v, method, BC, init, bathymetry=""):
    fileprefix = testdir + "/" + method + "_" + BC + "_" + init + "_" + bathymetry
    np.savetxt(fileprefix + "eta.dat", eta)
    np.savetxt(fileprefix + "u.dat", u)
    np.savetxt(fileprefix + "v.dat", v)

def loadResults(method, BC, init, bathymetry=""):
    fileprefix = testdir + "/" + method + "_" + BC + "_" + init + "_" + bathymetry
    eta = np.loadtxt(fileprefix + "eta.dat")
    u =   np.loadtxt(fileprefix + "u.dat")
    v =   np.loadtxt(fileprefix + "v.dat")
    return eta, u, v


def assertListAlmostEqual(theself, list1, list2, tol, testname):
    l = max(len(list1), len(list2))
    outro = ""
    if l < 6:
        outro = "\n\n- " + str(list1) + "\n+ " + str(list2)
        
    strList1 = str(list1)[:21]
    if (len(strList1) > 20):
        strList1 = strList1[:20] + "..."
    strList2 = str(list2)[:21]
    if (len(strList2) > 20):
        strList2 = strList2[:20] + "..."
        
    msg = "test case \'" + testname + "\' - lists differs: " + strList1 + " != " + strList2 + "\n\n"
    theself.assertEqual(len(list1), len(list2),
                     msg=msg + "Not same lengths:\nlen(list1) = " + str(len(list1)) + "\nlen(list2) = " + str(len(list2)) + outro)
    
    l = len(list1)
    outro = ""
    if l < 6:
        outro = "\n\n- " + str(list1) + "\n+ " + str(list2)
    i = 0
    for a,b in zip(list1, list2):
        theself.assertAlmostEqual(a, b, tol,
                               msg = msg + "First differing element " + str(i) + ":\n" + str(a) + "\n" + str(b) + outro)
        i = i + 1

def assert2DListAlmostEqual(theself, list1, list2, tol, testname):
    l = max(len(list1), len(list2))
    outro = ""
    if l < 6:
        outro = "\n\n- " + str(list1) + "\n+ " + str(list2)
        
    strList1 = str(list1)[:21]
    if (len(strList1) > 20):
        strList1 = strList1[:20] + "..."
    strList2 = str(list2)[:21]
    if (len(strList2) > 20):
        strList2 = strList2[:20] + "..."
        
    msg = "test case \'" + testname + "\' - lists differs: " + strList1 + " != " + strList2 + "\n\n"
    theself.assertEqual(len(list1), len(list2),
                     msg=msg + "Not same lengths:\nlen(list1) = " + str(len(list1)) + "\nlen(list2) = " + str(len(list2)) + outro)
    
    l = len(list1)
    outro = ""
    if l < 6:
        outro = "\n\n- " + str(list1) + "\n+ " + str(list2)
    i = 0
    for pos1, pos2 in zip(list1, list2):
        j = 0
        for a, b in zip(pos1, pos2):
            theself.assertAlmostEqual(a, b, tol,
                                      msg = msg + "First differing element " + str((i,j)) + ":\n" + str(a) + "\n" + str(b) + outro)
            j = j + 1
        i = i + 1


def assert2DListNotAlmostEqual(theself, list1, list2, tol, testname):
    l = max(len(list1), len(list2))
    outro = ""
    if l < 6:
        outro = "\n\n- " + str(list1) + "\n+ " + str(list2)
        
    strList1 = str(list1)[:21]
    if (len(strList1) > 20):
        strList1 = strList1[:20] + "..."
    strList2 = str(list2)[:21]
    if (len(strList2) > 20):
        strList2 = strList2[:20] + "..."
        
    msg = "test case \'" + testname + "\' - lists differs: " + strList1 + " != " + strList2 + "\n\n"
    theself.assertEqual(len(list1), len(list2),
                     msg=msg + "Not same lengths:\nlen(list1) = " + str(len(list1)) + "\nlen(list2) = " + str(len(list2)) + outro)
    
    l = len(list1)
    outro = ""
    if l < 6:
        outro = "\n\n- " + str(list1) + "\n+ " + str(list2)
    i = 0
    for pos1, pos2 in zip(list1, list2):
        j = 0
        for a, b in zip(pos1, pos2):
            theself.assertNotAlmostEqual(a, b, tol,
                                         msg = msg + "First non-differing element " + str((i,j)) + ":\n" + str(a) + "\n" + str(b) + outro)
            j = j + 1
        i = i + 1

        
def setNpRandomSeed():
    np.random.seed(1)
