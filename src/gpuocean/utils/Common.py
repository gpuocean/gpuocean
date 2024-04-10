# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

This python module implements the different helper functions and 
classes that are shared through out all elements of the code.

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

import numpy as np
import time
import signal
import subprocess
import tempfile
import sys
import re
import io
import hashlib
import logging
import gc

import pycuda
import pycuda.compiler as cuda_compiler
import pycuda.gpuarray
import pycuda.driver as cuda

import warnings
import functools





class Timer(object):
    """
    Class which keeps track of time spent for a section of code
    """
    def __init__(self, tag, log_level=logging.DEBUG):
        self.tag = tag
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000 # millisecs
        self.logger.log(self.log_level, "%s: %f ms", self.tag, self.msecs)

class PopenFileBuffer(object):
    """
    Simple class for holding a set of tempfiles
    for communicating with a subprocess
    """
    def __init__(self):
        self.stdout = tempfile.TemporaryFile(mode='w+t')
        self.stderr = tempfile.TemporaryFile(mode='w+t')

    def __del__(self):
        self.stdout.close()
        self.stderr.close()

    def read(self):
        self.stdout.seek(0)
        cout = self.stdout.read()
        self.stdout.seek(0, 2)

        self.stderr.seek(0)
        cerr = self.stderr.read()
        self.stderr.seek(0, 2)

        return cout, cerr

class IPEngine(object):
    """
    Class for starting IPEngines for MPI processing in IPython
    """
    def __init__(self, n_engines):
        self.logger = logging.getLogger(__name__)
        
        #Start ipcontroller
        self.logger.info("Starting IPController")
        self.c_buff = PopenFileBuffer()
        c_cmd = ["ipcontroller",  "--ip='*'"]
        c_params = dict()
        c_params['stderr'] = self.c_buff.stderr
        c_params['stdout'] = self.c_buff.stdout
        c_params['shell'] = False
        if os.name == 'nt':
            c_params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.c = subprocess.Popen(c_cmd, **c_params)
        
        #Wait until controller is running
        time.sleep(3)
        
        #Start engines
        self.logger.info("Starting IPEngines")
        self.e_buff = PopenFileBuffer()
        e_cmd = ["mpiexec", "-n", str(n_engines), "ipengine", "--mpi"]
        e_params = dict()
        e_params['stderr'] = self.e_buff.stderr
        e_params['stdout'] = self.e_buff.stdout
        e_params['shell'] = False
        if os.name == 'nt':
            e_params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.e = subprocess.Popen(e_cmd, **e_params)

        # attach to a running cluster
        import ipyparallel
        self.cluster = ipyparallel.Client()#profile='mpi')
        time.sleep(3)
        while(len(self.cluster.ids) != n_engines):
            time.sleep(0.5)
            self.logger.info("Waiting for cluster...")
            self.cluster = ipyparallel.Client()#profile='mpi')
        
        self.logger.info("Done")
        
    def __del__(self):
        self.shutdown()
    
    def shutdown(self):
        if (self.e is not None):
            if (os.name == 'nt'):
                self.logger.warn("Sending CTRL+C to IPEngine")
                self.e.send_signal(signal.CTRL_C_EVENT)
                
            try:
                self.e.communicate(timeout=3)
                self.e.kill()
            except subprocess.TimeoutExpired:
                self.logger.warn("Killing IPEngine")
                self.e.kill()
                self.e.communicate()
            self.e = None
                
            cout, cerr = self.e_buff.read()
            self.logger.info("IPEngine cout: {:s}".format(cout))
            self.logger.info("IPEngine cerr: {:s}".format(cerr))
            self.e_buff = None
            
            gc.collect()
            
        if (self.c is not None):
            if (os.name == 'nt'):
                self.logger.warn("Sending CTRL+C to IPController")
                self.c.send_signal(signal.CTRL_C_EVENT)
                
            try:
                self.c.communicate(timeout=3)
                self.c.kill()
            except subprocess.TimeoutExpired:
                self.logger.warn("Killing IPController")
                self.c.kill()
                self.c.communicate()
            self.c = None
                
            cout, cerr = self.c_buff.read()
            self.logger.info("IPController cout: {:s}".format(cout))
            self.logger.info("IPController cerr: {:s}".format(cerr))
            self.c_buff = None
        
            gc.collect()
        
    def elapsed(self):
        return time.time() - self.start
            
        
        
        
class ProgressPrinter(object):
    """
    Small helper class for 
    """
    def __init__(self, print_every=5):
        self.logger = logging.getLogger(__name__)
        self.start = time.time()
        self.print_every = print_every
        self.next_print_time = print_every
        self.print_string = ProgressPrinter.formatString(0, 0, 0)
        self.last_x = 0
        self.secs_per_iter = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass
        
    def getPrintString(self, x):
        elapsed =  time.time() - self.start
        
        if (x == 0):
            self.print_string = ProgressPrinter.formatString(0, 0, np.nan)
            return self.print_string
        
        if (elapsed >= self.next_print_time or x == 1.0):
            dt = elapsed - (self.next_print_time - self.print_every)
            dx = x - self.last_x
                        
            if (dt <= 0):
                return
                
            self.last_x = x
            self.next_print_time = max(elapsed, self.next_print_time + self.print_every)
            
            # A kind of floating average
            if not self.secs_per_iter:
                self.secs_per_iter = dt / dx
            self.secs_per_iter = 0.2*self.secs_per_iter + 0.8*(dt / dx)
            
            remaining_time = (1-x) * self.secs_per_iter
            
            self.print_string = ProgressPrinter.formatString(x, elapsed, remaining_time)
            
        return self.print_string
            

    def formatString(t, elapsed, remaining_time):
        return "{:s}. Total: {:s}, elapsed: {:s}, remaining: {:s}".format(
            ProgressPrinter.progressBar(t), 
            ProgressPrinter.timeString(elapsed + remaining_time), 
            ProgressPrinter.timeString(elapsed), 
            ProgressPrinter.timeString(remaining_time))
                

    def timeString(seconds):
        if np.isnan(seconds):
            return(str(seconds))
        
        seconds = int(max(seconds, 0))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        periods = [('h', hours), ('m', minutes), ('s', seconds + sys.float_info.epsilon)]
        time_string = ' '.join('{:d}{:s}'.format(int(value), name)
                                for name, value in periods
                                if value)
        return time_string

    def progressBar(t, width=30):
        progress = int(round(width * t))
        progressbar = "0% [" + "#"*(progress) + "="*(width-progress) + "] 100%"
        return progressbar


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated.
    Reference: https://stackoverflow.com/questions/2536307/how-do-i-deprecate-python-functions
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # MLS: Seems wrong to mess with standard filter settings in this context.
        # Code is nevertheless not removed, as this could be relevant at a later time.
        #warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        #warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

class CUDAContext(object):
    """
    Class which keeps track of the CUDA context and some helper functions
    """
    def __init__(self, device=0, blocking=False, use_cache=True):
        self.blocking = blocking
        self.use_cache = use_cache
        self.device = device
        self.logger =  logging.getLogger(__name__)
        self.kernels = {}
        
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        
        #Initialize cuda (must be first call to PyCUDA)
        cuda.init(flags=0)
        
        self.logger.info("PyCUDA version %s", str(pycuda.VERSION_TEXT))
        
        # Check that the device id is valid
        if self.device >= cuda.Device.count():
            self.device = self.device % cuda.Device.count()
            print('Changing device ID from ' + str(device) + ' to ' + str(self.device))
        
        #Print some info about CUDA
        self.logger.info("CUDA version %s", str(cuda.get_version()))
        self.logger.info("Driver version %s",  str(cuda.get_driver_version()))

        self.cuda_device = cuda.Device(self.device)
        self.logger.info("Using '%s' GPU", self.cuda_device.name())
        self.logger.debug(" => compute capability: %s", str(self.cuda_device.compute_capability()))
        self.logger.debug(" => memory: %d MB", self.cuda_device.total_memory() / (1024*1024))

        # Create the CUDA context
        if (self.blocking):
            self.cuda_context = self.cuda_device.make_context(flags=cuda.ctx_flags.SCHED_BLOCKING_SYNC)
            self.logger.warning("Using blocking context")
        else:
            self.cuda_context = self.cuda_device.make_context(flags=cuda.ctx_flags.SCHED_AUTO)
        
        self.logger.info("Created context handle <%s>", str(self.cuda_context.handle))

        #Create cache dir for cubin files
        if (self.use_cache):
            self.cache_path = os.path.join(self.module_path, "cuda_cache") 
            if not os.path.isdir(self.cache_path):
                os.mkdir(self.cache_path)
            self.logger.debug("Using CUDA cache dir %s", self.cache_path)
            
    def __del__(self, *args):
        self.logger.info("Cleaning up CUDA context handle <%s>", str(self.cuda_context.handle))
            
        # Loop over all contexts in stack, and remove "this"
        other_contexts = []
        while (cuda.Context.get_current() != None):
            context = cuda.Context.get_current()
            if (context.handle != self.cuda_context.handle):
                self.logger.debug("<%s> Popping <%s> (*not* ours)", str(self.cuda_context.handle), str(context.handle))
                other_contexts = [context] + other_contexts
                cuda.Context.pop()
            else:
                self.logger.debug("<%s> Popping <%s> (ours)", str(self.cuda_context.handle), str(context.handle))
                cuda.Context.pop()

        # Add all the contexts we popped that were not our own
        for context in other_contexts:
            self.logger.debug("<%s> Pushing <%s>", str(self.cuda_context.handle), str(context.handle))
            cuda.Context.push(context)
            
        self.logger.debug("<%s> Detaching", str(self.cuda_context.handle))
        self.cuda_context.detach()
        
    def __str__(self):
        return "CudaContext id " + str(self.cuda_context.handle)

    @staticmethod
    def hash_kernel(kernel_filename, include_dirs):        
        # Generate a kernel ID for our caches
        num_includes = 0
        max_includes = 100
        kernel_hasher = hashlib.md5()
        logger = logging.getLogger(__name__)
        
        # Loop over file and includes, and check if something has changed
        files = [kernel_filename]
        while len(files):
        
            if (num_includes > max_includes):
                raise("Maximum number of includes reached - circular include in {:}?".format(kernel_filename))
        
            filename = files.pop()
            
            logger.debug("Hashing %s", filename)
                
            modified = os.path.getmtime(filename)
                
            # Open the file
            with io.open(filename, "r") as file:
            
                # Search for #inclue <something> and also hash the file
                file_str = file.read()
                kernel_hasher.update(file_str.encode('utf-8'))
                kernel_hasher.update(str(modified).encode('utf-8'))
                
                #Find all includes
                includes = re.findall('^\W*#include\W+(.+?)\W*$', file_str, re.M)
                
            # Loop over everything that looks like an include
            for include_file in includes:
                
                #Search through include directories for the file
                file_path = os.path.dirname(filename)
                for include_path in [file_path] + include_dirs:
                
                    # If we find it, add it to list of files to check
                    temp_path = os.path.join(include_path, include_file)
                    if (os.path.isfile(temp_path)):
                        files = files + [temp_path]
                        num_includes = num_includes + 1 #For circular includes...
                        break
            
        return kernel_hasher.hexdigest()
        
    """
    Reads a text file and creates an CUDA kernel from that
    """
    def get_kernel(self, kernel_filename, include_dirs=[], defines={}, compile_args={'no_extern_c': True}, jit_compile_args={},
                   is_abs_path=False):
        """
        Helper function to print compilation output
        """
        def cuda_compile_message_handler(compile_success_bool, info_str, error_str):
            self.logger.debug("Compilation returned %s", str(compile_success_bool))
            if info_str:
                self.logger.debug("Info: %s", info_str)
            if error_str:
                self.logger.debug("Error: %s", error_str)
        
        self.logger.debug("Getting %s", kernel_filename)
        
        if not 'arch' in compile_args.keys():
            # HACK: Since CUDA 11.1 does not know about newer compute architectures that 8.6
            if ((cuda.Device(self.device).get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR) > 8) or
                (cuda.Device(self.device).get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR) == 8  and
                cuda.Device(self.device).get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR) > 6) ):
                compile_args['arch'] = "sm_80"
            
        if not 'arch' in compile_args.keys():
            # HACK: Since CUDA 11.1 does not know about newer compute architectures that 8.6
            if ((cuda.Device(self.device).get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR) > 8) or 
                (cuda.Device(self.device).get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR) == 8  and 
                cuda.Device(self.device).get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR) > 6) ):

                compile_args['arch'] = "sm_80"

        # Create a hash of the kernel (and its includes)
        options_hasher = hashlib.md5()
        options_hasher.update(str(defines).encode('utf-8') + str(compile_args).encode('utf-8'));
        options_hash = options_hasher.hexdigest()
        options_hasher = None
        root, ext = os.path.splitext(kernel_filename)
        _, filename_with_ext = os.path.split(kernel_filename)
        filename_no_ext, _ = os.path.splitext(filename_with_ext)
        kernel_path = kernel_filename
        if not is_abs_path:
            kernel_path = os.path.abspath(os.path.join(self.module_path, "..", "gpu_kernels", kernel_filename))
        hash = CUDAContext.hash_kernel( \
                    kernel_path, \
                    include_dirs=[os.path.join(self.module_path, "../kernels")] + include_dirs)
        kernel_hash = filename_no_ext + "_" +  hash + "_" + options_hash + ext
        cached_kernel_filename = os.path.join(self.cache_path, kernel_hash)
        
        # If we have the kernel in our hashmap, return it
        if (kernel_hash in self.kernels.keys()):
            self.logger.debug("Found kernel %s cached in hashmap (%s)", kernel_filename, kernel_hash)
            return self.kernels[kernel_hash]
        
        # If we have it on disk, return it
        elif (self.use_cache and os.path.isfile(cached_kernel_filename)):
            self.logger.debug("Found kernel %s cached on disk (%s)", kernel_filename, kernel_hash)
                
            with io.open(cached_kernel_filename, "rb") as file:
                file_str = file.read()
                module = cuda.module_from_buffer(file_str, message_handler=cuda_compile_message_handler, **jit_compile_args)
                
            self.kernels[kernel_hash] = module
            return self.kernels[kernel_hash]
            
        # Otherwise, compile it from source
        else:
            self.logger.debug("Compiling %s (%s)", kernel_filename, kernel_hash)
                
            #Create kernel string
            kernel_string = ""
            for key, value in defines.items():
                kernel_string += "#define {:s} {:s}\n".format(str(key), str(value))
            kernel_string += '#include "{:s}"'.format(str(kernel_path))
            if (self.use_cache):
                with io.open(cached_kernel_filename + ".txt", "w") as file:
                    #Why is kernel_string a bytes object in Python 3.5.2?
                    #Bugfix here
                    if isinstance(kernel_string, bytes):
                        kernel_string = bytes.decode(kernel_string)
                    file.write(kernel_string)
                
            
            with Timer("compiler") as timer:
                cubin = cuda_compiler.compile(kernel_string, include_dirs=include_dirs, cache_dir=False, **compile_args)
                module = cuda.module_from_buffer(cubin, message_handler=cuda_compile_message_handler, **jit_compile_args)
                if (self.use_cache):
                    with io.open(cached_kernel_filename, "wb") as file:
                        file.write(cubin)
                
            self.kernels[kernel_hash] = module
            
            return self.kernels[kernel_hash]

    def clear_kernel_cache(self):
        """
        Clears the kernel cache (useful for debugging & development)
        """
        self.logger.debug("Clearing cache")
        self.kernels = {}
        gc.collect()
        
    """
    Synchronizes all streams etc
    """
    def synchronize(self):
        self.cuda_context.synchronize()
        
        
        
class CUDAArray2D:
    """
    Class that holds data 
    """

    def __init__(self, gpu_stream, nx, ny, halo_x, halo_y, data, \
                 asymHalo=None, double_precision=False, integers=False):
        """
        Uploads initial data to the CUDA device
        """
        self.double_precision = double_precision
        self.integers = integers
        host_data = None
        if self.double_precision:
            host_data = data
        else:
            host_data = self.convert_to_float32(data)
        
        self.nx = nx
        self.ny = ny
        self.nx_halo = nx + 2*halo_x
        self.ny_halo = ny + 2*halo_y
        if (asymHalo is not None and len(asymHalo) == 4):
            # asymHalo = [halo_north, halo_east, halo_south, halo_west]
            self.nx_halo = nx + asymHalo[1] + asymHalo[3]
            self.ny_halo = ny + asymHalo[0] + asymHalo[2]
            
        assert(host_data.shape[1] == self.nx_halo), str(host_data.shape[1]) + " vs " + str(self.nx_halo)
        assert(host_data.shape[0] == self.ny_halo), str(host_data.shape[0]) + " vs " + str(self.ny_halo)
        
        assert(data.shape == (self.ny_halo, self.nx_halo))

        #Upload data to the device
        self.data = pycuda.gpuarray.to_gpu_async(host_data, stream=gpu_stream)
        self.holds_data = True

        self.mask = None
        if (np.ma.is_masked(data)):
            self.mask = data.mask
        
        self.bytes_per_float = host_data.itemsize
        assert(self.bytes_per_float == 4 or
              (self.double_precision and self.bytes_per_float == 8))
        self.pitch = np.int32((self.nx_halo)*self.bytes_per_float)
        
        
    def upload(self, gpu_stream, data):
        """
        Filling the allocated buffer with new data
        """
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before upload is called')
        
        if (np.ma.is_masked(data)):
            self.mask = data.mask

        # Make sure that the input is of correct size:
        if self.double_precision:
            host_data = data
        else:
            host_data = self.convert_to_float32(data)
            
        assert(host_data.shape[1] == self.nx_halo), str(host_data.shape[1]) + " vs " + str(self.nx_halo)
        assert(host_data.shape[0] == self.ny_halo), str(host_data.shape[0]) + " vs " + str(self.ny_halo)
        assert(host_data.shape == (self.ny_halo, self.nx_halo))
        
        assert(host_data.itemsize == self.bytes_per_float), "Host data itemsize is " + str(host_data.itemsize) + ", but should have been " + str(self.bytes_per_float)
        
        # Okay, everything is fine, now upload:
        self.data.set_async(host_data, stream=gpu_stream)
        
    
    def copyBuffer(self, gpu_stream, buffer):
        """
        Copying the given device buffer into the already allocated memory
        """
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before copying buffer')
        
        if not buffer.holds_data:
            raise RuntimeError('The provided buffer is either not allocated, or has been freed before copying buffer')
        
        # Make sure that the input is of correct size:
        assert(buffer.nx_halo == self.nx_halo), str(buffer.nx_halo) + " vs " + str(self.nx_halo)
        assert(buffer.ny_halo == self.ny_halo), str(buffer.ny_halo) + " vs " + str(self.ny_halo)
        
        assert(buffer.bytes_per_float == self.bytes_per_float), "Provided buffer itemsize is " + str(buffer.bytes_per_float) + ", but should have been " + str(self.bytes_per_float)
        
        # Okay, everything is fine - issue device-to-device-copy:
        total_num_bytes = self.bytes_per_float*self.nx_halo*self.ny_halo
        cuda.memcpy_dtod_async(self.data.ptr, buffer.data.ptr, total_num_bytes, stream=gpu_stream)
        
        
        
    def download(self, gpu_stream):
        """
        Enables downloading data from CUDA device to Python
        """
        if not self.holds_data:
            raise RuntimeError('CUDA buffer has been freed')
        
        #Copy data from device to host
        host_data = self.data.get(stream=gpu_stream)

        if (self.mask is not None):
            host_data = np.ma.array(host_data, mask=self.mask)
        
        #Return
        return host_data

    
    def release(self):
        """
        Frees the allocated memory buffers on the GPU 
        """
        if self.holds_data:
            self.data.gpudata.free()
            self.holds_data = False

    
    @staticmethod
    def convert_to_float32(data):
        """
        Converts to C-style float 32 array suitable for the GPU/CUDA
        """
        if (not np.issubdtype(data.dtype, np.float32) or np.isfortran(data)):
            return data.astype(np.float32, order='C')
        else:
            return data

    
class SWEDataArakawaA:
    """
    A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
    """

    def __init__(self, gpu_stream, nx, ny, halo_x, halo_y, h0, hu0, hv0):
        """
        Uploads initial data to the CUDA device
        """
        self.h0  = CUDAArray2D(gpu_stream, nx, ny, halo_x, halo_y, h0)
        self.hu0 = CUDAArray2D(gpu_stream, nx, ny, halo_x, halo_y, hu0)
        self.hv0 = CUDAArray2D(gpu_stream, nx, ny, halo_x, halo_y, hv0)
        
        self.h1  = CUDAArray2D(gpu_stream, nx, ny, halo_x, halo_y, h0)
        self.hu1 = CUDAArray2D(gpu_stream, nx, ny, halo_x, halo_y, hu0)
        self.hv1 = CUDAArray2D(gpu_stream, nx, ny, halo_x, halo_y, hv0)

    def swap(self):
        """
        Swaps the variables after a timestep has been completed
        """
        self.h1,  self.h0  = self.h0,  self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1
        
    def download(self, gpu_stream):
        """
        Enables downloading data from CUDA device to Python
        """
        h_cpu  = self.h0.download(gpu_stream)
        hu_cpu = self.hu0.download(gpu_stream)
        hv_cpu = self.hv0.download(gpu_stream)
        
        return h_cpu, hu_cpu, hv_cpu
    

    def downloadPrevTimestep(self, gpu_stream):
        """
        Enables downloading data from the additional buffer of CUDA device to Python
        """
        h_cpu  = self.h1.download(gpu_stream)
        hu_cpu = self.hu1.download(gpu_stream)
        hv_cpu = self.hv1.download(gpu_stream)
        
        return h_cpu, hu_cpu, hv_cpu

    def release(self):
        """
        Frees the allocated memory buffers on the GPU 
        """
        self.h0.release()
        self.hu0.release()
        self.hv0.release()
        self.h1.release()
        self.hu1.release()
        self.hv1.release()
        
        
        
        
class SWEDataArakawaC:
    """
    A class representing an Arakawa C type (staggered, u fluxes on east/west faces, v fluxes on north/south faces) grid
    We use h as cell centers
    """
    def __init__(self, gpu_stream, nx, ny, halo_x, halo_y, h0, hu0, hv0, \
                 fbl=False):
        """
        Uploads initial data to the CUDA device
        asymHalo needs to be on the form [north, east, south, west]
        """
        #FIXME: This at least works for 0 and 1 ghost cells, but not convinced it generalizes
        assert(halo_x <= 1 and halo_y <= 1)
        
        self.fbl = fbl
        
        if (fbl):
            self.h0   = CUDAArray2D(gpu_stream, nx  , ny  , halo_x, halo_y, h0)
            self.hu0  = CUDAArray2D(gpu_stream, nx-1, ny  , halo_x, halo_y, hu0)
            self.hv0  = CUDAArray2D(gpu_stream, nx  , ny+1, halo_x, halo_y, hv0)
                                                          
            self.h1   = CUDAArray2D(gpu_stream, nx  , ny  , halo_x, halo_y, h0)
            self.hu1  = CUDAArray2D(gpu_stream, nx-1, ny  , halo_x, halo_y, hu0)
            self.hv1  = CUDAArray2D(gpu_stream, nx  , ny+1, halo_x, halo_y, hv0)
            
            return

        #print "SWEDataArakawaC"
        #print "(h0.shape, (nx, ny), asymHalo,  (halo_x, halo_y)): ", (h0.shape, (nx, ny), asymHalo,  (halo_x, halo_y))
        #print "(hu0.shape, (nx, ny), asymHalo, (halo_x, halo_y)): ", (hu0.shape, (nx+1, ny), asymHaloU,  (halo_x, halo_y))
        #print "(hv0.shape, (nx, ny), asymHalo,  (halo_x, halo_y)): ", (hv0.shape, (nx, ny+1), asymHaloV, (halo_x, halo_y))

        self.h0   = CUDAArray2D(gpu_stream, nx, ny, halo_x, halo_y, h0)
        self.hu0  = CUDAArray2D(gpu_stream, nx+1, ny, halo_x, halo_y, hu0)
        self.hv0  = CUDAArray2D(gpu_stream, nx, ny+1, halo_x, halo_y, hv0)
        
        self.h1   = CUDAArray2D(gpu_stream, nx, ny, halo_x, halo_y, h0)
        self.hu1  = CUDAArray2D(gpu_stream, nx+1, ny, halo_x, halo_y, hu0)
        self.hv1  = CUDAArray2D(gpu_stream, nx, ny+1, halo_x, halo_y, hv0)
                   
        
    def swap(self):
        """
        Swaps the variables after a timestep has been completed
        """
        #h is assumed to be constant (bottom topography really)
        self.h1,  self.h0  = self.h0, self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1
        
    def download(self, gpu_stream, interior_domain_only=False):
        """
        Enables downloading data from CUDA device to Python
        """
        h_cpu  = self.h0.download(gpu_stream)
        hu_cpu = self.hu0.download(gpu_stream)
        hv_cpu = self.hv0.download(gpu_stream)
        
        if (interior_domain_only and self.fbl):
            #print("Sneaking in some FBL specific functionality")
            return h_cpu[1:-1, 1:-1], hu_cpu[1:-1,:], hv_cpu[1:-1, 1:-1]
        
        return h_cpu, hu_cpu, hv_cpu

    def downloadPrevTimestep(self, gpu_stream):
        """
        Enables downloading data from the additional buffer of CUDA device to Python
        """
        h_cpu  = self.h1.download(gpu_stream)
        hu_cpu = self.hu1.download(gpu_stream)
        hv_cpu = self.hv1.download(gpu_stream)
        
        return h_cpu, hu_cpu, hv_cpu
    
    def release(self):
        """
        Frees the allocated memory buffers on the GPU 
        """
        self.h0.release()
        self.hu0.release()
        self.hv0.release()
        self.h1.release()
        self.hu1.release()
        self.hv1.release()


class BoundaryConditions:
    """
    Class that represents different forms for boundary conidtions
    """
    def __init__(self, \
                 north=1, east=1, south=1, west=1, \
                 spongeCells={'north':0, 'south': 0, 'east': 0, 'west': 0}):
        """
        There is one parameter for each of the cartesian boundaries.
        Values can be set as follows:
        1 = Wall
        2 = Periodic (requires same for opposite boundary as well)
        3 = Open Boundary with Flow Relaxation Scheme
        4 = Open linear interpolation
        Options 3 and 4 are of sponge type (requiring extra computational domain)
        """
        self.north = np.int32(north)
        self.east  = np.int32(east)
        self.south = np.int32(south)
        self.west  = np.int32(west)
        self.spongeCells = spongeCells
        
        #Quick and dirty make sure sponge cells are int32
        for key in self.spongeCells.keys():
            self.spongeCells[key] = np.int32(self.spongeCells[key])
            
        # Checking that periodic boundaries are periodic
        assert not ((self.north == 2 or self.south == 2) and  \
                    (self.north != self.south)), \
                    'The given periodic boundary conditions are not periodically (north/south)'
        assert not ((self.east == 2 or self.west == 2) and \
                    (self.east != self.west)), \
                    'The given periodic boundary conditions are not periodically (east/west)'

    def get(self):
        raise NotImplementedError("get() implementation changed! Change calling code")
        #Previous code was:
        #return [self.north, self.east, self.south, self.west]
        #New code is the following, change code calling this function to support new dict
        return {'north': self.north, 'south': self.south, 'east': self.east, 'west': self.west}
    
    def getBCDict(self):
        return {'north': self.north, 'south': self.south, 'east': self.east, 'west': self.west}
    
    def getSponge(self):
        return self.spongeCells
    
    def isDefault(self):
        return (self.north == 1 and \
                self.east == 1 and \
                self.south == 1 and \
                self.east == 1)

    def isSponge(self):
        return (self.north == 3 or self.north == 4 or \
                self.east  == 3 or self.east  == 4 or \
                self.south == 3 or self.south == 4 or \
                self.west  == 3 or self.west  == 4)
    
    def isPeriodicNorthSouth(self):
        return (self.north == 2 and self.south == 2)
    
    def isPeriodicEastWest(self):
        return (self.east == 2 and self.west == 2)

    def isPeriodic(self):
        return (self.isPeriodicEastWest() and self.isPeriodicNorthSouth())
    
    
    def _toString(self, cond):
        if cond == 1:
            return "Wall"
        elif cond == 2:
            return "Periodic"
        elif cond == 3:
            return "Flow_Relaxation_Scheme"
        elif cond == 4:
            return "Open_Linear_Interpolation"
        else:
            return "Invalid_:|"
        
    def __str__(self):
        msg = "north: "   + self._toString(self.north) + \
              ", east: "  + self._toString(self.east)  + \
              ", south: " + self._toString(self.south) + \
              ", west: "  + self._toString(self.west)
        msg = msg + ", spongeCells: " + str(self.spongeCells)
        return msg
    
    
    @classmethod
    def fromstring(cls, bc_string):
    
        def keyword_to_cond(key):
            if key == 'Wall':
                return 1
            elif key == 'Periodic':
                return 2
            elif key == 'Flow_Relaxation_Scheme':
                return 3
            elif key == 'Open_Linear_Interpolation':
                return 4
            else:
                return -1
            
        #clean string
        bc_clean_str = bc_string.replace(',','').replace('}','').replace('{','').replace('[','').replace(']','').replace(':','').replace("'","")
        
        bc_array = str.split(bc_clean_str, ' ')
        north = keyword_to_cond(bc_array[1])
        east  = keyword_to_cond(bc_array[3])
        south = keyword_to_cond(bc_array[5])
        west  = keyword_to_cond(bc_array[7])
        
        if len(bc_array)>13:
            spongeCells = {bc_array[9]: int(bc_array[10]),
                            bc_array[11]:  int(bc_array[12]),
                            bc_array[13]:  int(bc_array[14]),
                            bc_array[15]:  int(bc_array[16])}
        else:
            spongeCells = {'north'  : int(bc_array[9]),
                            'east'  :  int(bc_array[10]),
                            'south' :  int(bc_array[11]),
                            'west'  :  int(bc_array[12])}
        
        return cls(north=north, east=east, south=south, west=west, spongeCells=spongeCells)
    
    
    
    
    
class SingleBoundaryConditionData():
    """
    This class holds the external solution for a single boundary over time.
    """
    def __init__(self, h=None, hu=None, hv=None):
        self.h = [np.zeros((1,2), dtype=np.float32, order='C')]
        self.hu = [np.zeros((1,2), dtype=np.float32, order='C')]
        self.hv = [np.zeros((1,2), dtype=np.float32, order='C')]
        self.numSteps = 1
        self.shape = self.h[0].shape
        
        if h is not None:
            self.shape = h[0].shape
            self.numSteps = len(h)
            
            self.h = h
            self.hu = hu
            self.hv = hv
        
            for i in range(len(h)):
                assert( h[i].shape == self.shape), str(self.shape) + " vs " + str(h[i].shape)
                assert(hu[i].shape == self.shape), str(self.shape) + " vs " + str(hu[i].shape)
                assert(hv[i].shape == self.shape), str(self.shape) + " vs " + str(hv[i].shape)
                
                assert (h[i].dtype == 'float32'), "h data needs to be of type np.float32"
                assert (hu[i].dtype == 'float32'), "hu data needs to be of type np.float32"
                assert (hv[i].dtype == 'float32'), "hv data needs to be of type np.float32"
                
    def __str__(self):
        return str(self.numSteps) + " steps, each " + str(self.shape)
    
class BoundaryConditionsData():
    """
    This class holds external solution for all boundaries over time.
    """
    
    def __init__(self, 
                    t=None, \
                    north=SingleBoundaryConditionData(), \
                    south=SingleBoundaryConditionData(), \
                    east=SingleBoundaryConditionData(), \
                    west=SingleBoundaryConditionData()):

        self.t = [0]
        self.numSteps = 1
        self.north = north
        self.south = south
        self.east = east
        self.west = west
                
        if t is not None:
            self.t = t
            self.numSteps = len(t)
            
        for data in [north, south, east, west]:
            assert(data.numSteps == self.numSteps), "Wrong timesteps " + str(data.numSteps) + " vs " + str(self.numSteps)
            
        assert (north.h[0].shape == south.h[0].shape), "Wrong shape of north vs south " + str(north.h[0].shape) + " vs " + str(south.h[0].shape)
        assert (east.h[0].shape == west.h[0].shape), "Wrong shape of east vs west " + str(east.h[0].shape) + " vs " + str(west.h[0].shape)

    def __str__(self):
        return "Steps=" + str(self.numSteps) \
            + ", [north=" + str(self.north) + "]" \
            + ", [south=" + str(self.south) + "]" \
            + ", [east=" + str(self.east) + "]" \
            + ", [west=" + str(self.west) + "]"
    
    
    
    
class BoundaryConditionsArakawaA:
    """
    Class that checks boundary conditions and calls the required kernels for Arakawa A type grids.
    """

    def __init__(self, gpu_ctx, nx, ny, \
                 halo_x, halo_y, \
                 boundary_conditions, \
                 bc_data=BoundaryConditionsData(), \
                 block_width = 16, block_height = 16):
        self.logger = logging.getLogger(__name__)

        self.boundary_conditions = boundary_conditions
        
        self.nx = np.int32(nx) 
        self.ny = np.int32(ny) 
        self.halo_x = np.int32(halo_x)
        self.halo_y = np.int32(halo_y)
        self.bc_data = bc_data;
        #print("boundary (ny, nx: ", (self.ny, self.nx))
        #print("boundary (halo_y, halo_x): ", (self.halo_y, self.halo_x))
        #print("numerical sponge cells (n,e,s,w): ", self.boundary_conditions.spongeCells)
        
        # Load CUDA module for periodic boundary
        self.boundaryKernels = gpu_ctx.get_kernel("boundary_kernels.cu", defines={'block_width': block_width, 'block_height': block_height})

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.periodicBoundary_NS = self.boundaryKernels.get_function("periodicBoundary_NS")
        self.periodicBoundary_NS.prepare("iiiiPiPiPi")
        self.periodicBoundary_EW = self.boundaryKernels.get_function("periodicBoundary_EW")
        self.periodicBoundary_EW.prepare("iiiiPiPiPi")
        self.linearInterpolation_NS = self.boundaryKernels.get_function("linearInterpolation_NS")
        self.linearInterpolation_NS.prepare("iiiiiiiiPiPiPi")
        self.linearInterpolation_EW = self.boundaryKernels.get_function("linearInterpolation_EW")
        self.linearInterpolation_EW.prepare("iiiiiiiiPiPiPi")
        self.flowRelaxationScheme_NS = self.boundaryKernels.get_function("flowRelaxationScheme_NS")
        self.flowRelaxationScheme_NS.prepare("iiiiiiiiPiPiPif")
        self.flowRelaxationScheme_EW = self.boundaryKernels.get_function("flowRelaxationScheme_EW")
        self.flowRelaxationScheme_EW.prepare("iiiiiiiiPiPiPif")
        
        self.bc_timestamps = [None, None]
        self.bc_textures = None
       
        # Set kernel launch parameters
        self.local_size = (block_width, block_height, 1)
        self.global_size = ( \
                             int(np.ceil((self.nx + 2*self.halo_x + 1)/float(self.local_size[0]))), \
                             int(np.ceil((self.ny + 2*self.halo_y + 1)/float(self.local_size[1]))) )


        
    """
    Function which updates the external solution for the boundary conditions
    """
    def update_bc_values(self, gpu_stream, t):
        #Only if we use flow relaxation
        if not (self.boundary_conditions.north == 3 or \
                self.boundary_conditions.south == 3 or \
                self.boundary_conditions.east == 3 or \
                self.boundary_conditions.west == 3):
            return
    
    
        #Compute new t0 and t1
        t_max_index = len(self.bc_data.t)-1
        t0_index = max(0, np.searchsorted(self.bc_data.t, t)-1)
        t1_index = min(t_max_index, np.searchsorted(self.bc_data.t, t))
        new_t0 = self.bc_data.t[t0_index]
        new_t1 = self.bc_data.t[t1_index]
        
        #Find the old (and update)
        old_t0 = self.bc_timestamps[0]
        old_t1 = self.bc_timestamps[1]
        self.bc_timestamps = [new_t0, new_t1]
        
        #Log some debug info
        self.logger.debug("Times: %s", str(self.bc_data.t))
        self.logger.debug("Time indices: [%d, %d]", t0_index, t1_index)
        self.logger.debug("Time: %s  New interval is [%s, %s], old was [%s, %s]", \
                    t, new_t0, new_t1, old_t0, old_t1)
                
        #Get texture references
        if (self.bc_textures):
            NS0_texref, NS1_texref, EW0_texref, EW1_texref = self.bc_textures;
        else:
            NS0_texref = self.boundaryKernels.get_texref("bc_tex_NS_current")
            EW0_texref = self.boundaryKernels.get_texref("bc_tex_EW_current")
            NS1_texref = self.boundaryKernels.get_texref("bc_tex_NS_next")
            EW1_texref = self.boundaryKernels.get_texref("bc_tex_EW_next")
        
        #Helper function to upload data to the GPU as a texture
        def setTexture(texref, numpy_array):       
            #Upload data to GPU and bind to texture reference
            #shape is interpreted as height, width, num_channels for order == “C”,
            data = np.ascontiguousarray(numpy_array)
            texref.set_array(cuda.make_multichannel_2d_array(data, order="C"))
            #cuda.bind_array_to_texref(cuda.make_multichannel_2d_array(numpy_array, order="C"), texref)
                        
            # Set texture parameters
            texref.set_filter_mode(cuda.filter_mode.LINEAR) #bilinear interpolation
            texref.set_address_mode(0, cuda.address_mode.CLAMP) #no indexing outside domain
            texref.set_address_mode(1, cuda.address_mode.CLAMP)
            texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES) #Use [0, 1] indexing
            
        def packDataNS(data, t_index):
            h = data.h[t_index]
            hu = data.hu[t_index]
            hv = data.hv[t_index]
            
            h = np.squeeze(h)
            hu = np.squeeze(hu)
            hv = np.squeeze(hv)
            zeros = np.zeros_like(h)
            assert(len(h.shape) == 1), "NS-data must be one row"
            nx = h.shape[0]
            
            components = 4
            NS_data = np.vstack((h, hu, hv, zeros))
            NS_data = np.transpose(NS_data);
            NS_data = np.reshape(NS_data, (1, nx, components))
            NS_data = np.ascontiguousarray(NS_data)
            #print(NS_data)
            
            return NS_data
            
        def packDataEW(data, t_index):
            h = data.h[t_index]
            hu = data.hu[t_index]
            hv = data.hv[t_index]
            
            h = np.squeeze(h)
            hu = np.squeeze(hu)
            hv = np.squeeze(hv)
            zeros = np.zeros_like(h)
            assert(len(h.shape) == 1), "EW-data must be one column"
            ny = h.shape[0]
            
            components = 4
            EW_data = np.vstack((h, hu, hv, zeros))
            EW_data = np.transpose(EW_data);
            EW_data = np.reshape(EW_data, (ny, 1, components))
            EW_data = np.ascontiguousarray(EW_data)
            
            return EW_data

            
        #If time interval has changed, upload new data
        if (new_t0 != old_t0):
            gpu_stream.synchronize()
            self.logger.debug("Updating T0")
            
            N_data = packDataNS(self.bc_data.north, t0_index)
            S_data = packDataNS(self.bc_data.south, t0_index)
            NS_data = np.vstack((S_data, N_data))
            setTexture(NS0_texref, NS_data)
            self.flowRelaxationScheme_NS.param_set_texref(NS0_texref)
            
            
            E_data = packDataEW(self.bc_data.east, t0_index)
            W_data = packDataEW(self.bc_data.west, t0_index)
            EW_data = np.hstack((W_data, E_data)) 
            setTexture(EW0_texref, EW_data)
            self.flowRelaxationScheme_EW.param_set_texref(EW0_texref)
            
            self.logger.debug("NS-Data is set to " + str(NS_data) + ", " + str(NS_data.shape))
            self.logger.debug("EW-Data is set to " + str(EW_data) + ", " + str(EW_data.shape))
            
            gpu_stream.synchronize()

        if (new_t1 != old_t1):
            gpu_stream.synchronize()
            self.logger.debug("Updating T1")
            
            N_data = packDataNS(self.bc_data.north, t1_index)
            S_data = packDataNS(self.bc_data.south, t1_index)
            NS_data = np.vstack((S_data, N_data))
            setTexture(NS1_texref, NS_data)
            self.flowRelaxationScheme_NS.param_set_texref(NS1_texref)
            
            E_data = packDataEW(self.bc_data.east, t1_index)
            W_data = packDataEW(self.bc_data.west, t1_index)
            EW_data = np.hstack((W_data, E_data)) 
            setTexture(EW1_texref, EW_data)
            self.flowRelaxationScheme_EW.param_set_texref(EW1_texref)
            
            self.logger.debug("NS-Data is set to " + str(NS_data) + ", " + str(NS_data.shape))
            self.logger.debug("EW-Data is set to " + str(EW_data) + ", " + str(EW_data.shape))
            
            gpu_stream.synchronize()
                
        # Store texture references (they are deleted if collected by python garbage collector)
        self.logger.debug("Textures: \n[%s, %s, %s, %s]", NS0_texref, NS1_texref, EW0_texref, EW1_texref)
        self.bc_textures = [NS0_texref, NS1_texref, EW0_texref, EW1_texref]
        
        # Update the bc_t linear interpolation coefficient
        elapsed_since_t0 = (t-new_t0)
        time_interval = max(1.0e-10, (new_t1-new_t0))
        self.bc_t = np.float32(max(0.0, min(1.0, elapsed_since_t0 / time_interval)))
        self.logger.debug("Interpolation t is %f", self.bc_t)
        
    def boundaryCondition(self, gpu_stream, h, u, v):
        if self.boundary_conditions.north == 2:
            self.periodic_boundary_NS(gpu_stream, h, u, v)
        else:
            if (self.boundary_conditions.north == 3 or \
                self.boundary_conditions.south == 3):
                self.flow_relaxation_NS(gpu_stream, h, u, v)
            if (self.boundary_conditions.north == 4 or \
                self.boundary_conditions.south == 4):
                self.linear_interpolation_NS(gpu_stream, h, u, v)
            
            
        if self.boundary_conditions.east == 2:
            self.periodic_boundary_EW(gpu_stream, h, u, v)
        else:
            if (self.boundary_conditions.east == 3 or \
                self.boundary_conditions.west == 3):
                self.flow_relaxation_EW(gpu_stream, h, u, v)
            if (self.boundary_conditions.east == 4 or \
                self.boundary_conditions.west == 4):
                self.linear_interpolation_EW(gpu_stream, h, u, v)
             
    def periodic_boundary_NS(self, gpu_stream, h, u, v):
        self.periodicBoundary_NS.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            h.data.gpudata, h.pitch, \
            u.data.gpudata, u.pitch, \
            v.data.gpudata, v.pitch)
        

    def periodic_boundary_EW(self, gpu_stream, h, v, u):
        self.periodicBoundary_EW.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            h.data.gpudata, h.pitch, \
            u.data.gpudata, u.pitch, \
            v.data.gpudata, v.pitch)


    def linear_interpolation_NS(self, gpu_stream, h, u, v):
        self.linearInterpolation_NS.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.boundary_conditions.north, self.boundary_conditions.south, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells['north'], \
            self.boundary_conditions.spongeCells['south'], \
            h.data.gpudata, h.pitch, \
            u.data.gpudata, u.pitch, \
            v.data.gpudata, v.pitch)                                   

    def linear_interpolation_EW(self, gpu_stream, h, u, v):
        self.linearInterpolation_EW.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.boundary_conditions.east, self.boundary_conditions.west, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells['east'], \
            self.boundary_conditions.spongeCells['west'], \
            h.data.gpudata, h.pitch, \
            u.data.gpudata, u.pitch, \
            v.data.gpudata, v.pitch)

    def flow_relaxation_NS(self, gpu_stream, h, u, v):
        self.flowRelaxationScheme_NS.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.boundary_conditions.north, self.boundary_conditions.south, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells['north'], \
            self.boundary_conditions.spongeCells['south'], \
            h.data.gpudata, h.pitch, \
            u.data.gpudata, u.pitch, \
            v.data.gpudata, v.pitch, \
            self.bc_t)

    def flow_relaxation_EW(self, gpu_stream, h, u, v):
        self.flowRelaxationScheme_EW.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.boundary_conditions.east, self.boundary_conditions.west, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells['east'], \
            self.boundary_conditions.spongeCells['west'], \
            h.data.gpudata, h.pitch, \
            u.data.gpudata, u.pitch, \
            v.data.gpudata, v.pitch, \
            self.bc_t)


        
class Bathymetry:
    """
    Class for holding bathymetry defined on cell intersections (cell corners) and reconstructed on 
    cell mid-points.
    """
    
    def __init__(self, gpu_ctx, gpu_stream, nx, ny, halo_x, halo_y, Bi_host, \
                 boundary_conditions=BoundaryConditions(), \
                 block_width=16, block_height=16):
        # Convert scalar data to int32
        self.gpu_stream = gpu_stream
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.halo_x = np.int32(halo_x)
        self.halo_y = np.int32(halo_y)
        self.halo_nx = np.int32(nx + 2*halo_x)
        self.halo_ny = np.int32(ny + 2*halo_y)
        self.boundary_conditions = boundary_conditions
        
        # Set land value (if masked array)
        self.mask_value = np.float32(1.0e20)
        self.use_mask = False
        if (np.ma.is_masked(Bi_host)):
            Bi_host = Bi_host.copy().filled(self.mask_value).astype(np.float32)
            self.use_mask = True
             
        # Check that Bi has the size corresponding to number of cell intersections
        BiShapeY, BiShapeX = Bi_host.shape
        assert(BiShapeX == nx+1+2*halo_x and BiShapeY == ny+1+2*halo_y), \
                "Wrong size of bottom bathymetry, should be defined on cell intersections, not cell centers. " + \
                str((BiShapeX, BiShapeY)) + " vs " + str((nx+1+2*halo_x, ny+1+2*halo_y))
        
        # Upload Bi to device
        self.Bi = CUDAArray2D(gpu_stream, nx+1, ny+1, halo_x, halo_y, Bi_host)

        # Define OpenCL parameters
        self.local_size = (block_width, block_height, 1) 
        self.global_size = ( \
                       int(np.ceil( (self.halo_nx+1) / float(self.local_size[0]))), \
                       int(np.ceil( (self.halo_ny+1) / float(self.local_size[1]))) \
        ) 
        
        # Check boundary conditions and make Bi periodic if necessary
        # Load CUDA module for periodic boundary
        self.boundaryKernels = gpu_ctx.get_kernel("boundary_kernels.cu", defines={'block_width': block_width, 'block_height': block_height})

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.periodic_boundary_intersections_NS = self.boundaryKernels.get_function("periodic_boundary_intersections_NS")
        self.periodic_boundary_intersections_NS.prepare("iiiiPi")
        self.periodic_boundary_intersections_EW = self.boundaryKernels.get_function("periodic_boundary_intersections_EW")
        self.periodic_boundary_intersections_EW.prepare("iiiiPi")
        self.closed_boundary_intersections_NS = self.boundaryKernels.get_function("closed_boundary_intersections_NS")
        self.closed_boundary_intersections_NS.prepare("iiiiPi")
        self.closed_boundary_intersections_EW = self.boundaryKernels.get_function("closed_boundary_intersections_EW")
        self.closed_boundary_intersections_EW.prepare("iiiiPi")

        self._boundaryConditions()
        
        # Allocate Bm
        Bm_host = np.zeros((self.halo_ny, self.halo_nx), dtype=np.float32, order='C')
        self.Bm = CUDAArray2D(gpu_stream, nx, ny, halo_x, halo_y, Bm_host)

        # Load kernel for finding Bm from Bi
        self.initBm_kernel = gpu_ctx.get_kernel("initHm_kernel.cu", defines={'block_width': block_width, 'block_height': block_height})

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.initBm = self.initBm_kernel.get_function("initBm")
        self.initBm.prepare("iiPifPi")
        self.waterElevationToDepth = self.initBm_kernel.get_function("waterElevationToDepth")
        self.waterElevationToDepth.prepare("iiPiPi")

        # Call kernel
        self.initBm.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                                   self.halo_nx, self.halo_ny, \
                                   self.Bi.data.gpudata, self.Bi.pitch, \
                                   self.mask_value, \
                                   self.Bm.data.gpudata, self.Bm.pitch)

                 
    def download(self, gpu_stream):
        Bm_cpu = self.Bm.download(gpu_stream)
        Bi_cpu = self.Bi.download(gpu_stream)

        #Mask land values in output
        if (self.use_mask):
            Bi_cpu = np.ma.array(data=Bi_cpu, mask=(Bi_cpu == self.mask_value), fill_value=0.0)
            Bm_cpu = np.ma.array(data=Bm_cpu, mask=(Bm_cpu == self.mask_value), fill_value=0.0)
            
        return Bi_cpu, Bm_cpu

    def release(self):
        """
        Frees the allocated memory buffers on the GPU 
        """
        self.Bm.release()
        self.Bi.release()
        
    # Transforming water elevation into water depth
    def waterElevationToDepth(self, h):
        
        assert ((h.ny_halo, h.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "h0 not the correct shape: " + str(h0.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))

        # Call kernel        
        self.waterElevationToDepth.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                                   self.halo_nx, self.halo_ny, \
                                   h.data.gpudata, h.pitch, \
                                   self.Bm.data.gpudata, self.Bm.pitch)
        
    # Transforming water depth into water elevation
    def waterDepthToElevation(self, w, h):

        assert ((h.ny_halo, h.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "h0 not the correct shape: " + str(h0.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))
        assert ((w.ny_halo, w.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "w not the correct shape: " + str(w.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))
        # Call kernel        
        self.waterDepthToElevation.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                                   self.halo_nx, self.halo_ny, \
                                   w.data.gpudata, w.pitch, \
                                   h.data.gpudata, h.pitch, \
                                   self.Bm.data.gpudata, self.Bm.pitch)
        
    def _boundaryConditions(self):
        # North-south:
        if (self.boundary_conditions.north == 2) and (self.boundary_conditions.south == 2):
            self.periodic_boundary_intersections_NS.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data.gpudata, self.Bi.pitch)
        elif (self.boundary_conditions.north == 1) and (self.boundary_conditions.south == 1):
            self.closed_boundary_intersections_NS.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data.gpudata, self.Bi.pitch)

        # East-west:
        if (self.boundary_conditions.east == 2) and (self.boundary_conditions.west == 2):
            self.periodic_boundary_intersections_EW.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data.gpudata, self.Bi.pitch)
        elif (self.boundary_conditions.north == 1) and (self.boundary_conditions.south == 1):
            self.closed_boundary_intersections_EW.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data.gpudata, self.Bi.pitch)
                 
                 
                 
                 
