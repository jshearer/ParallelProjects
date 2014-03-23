from numpy import version as numpy_version
from pycuda import VERSION as CUDAVERSION
from sys import version as PYVERSION
from tables import getPyTablesVersion, getHDF5Version
import shlex, subprocess
from string import join as sjoin
import re

"""
linux: godzilla@speggy:~/Desktop/Science/ParallelProjects/fractal$ uname -a
Linux K20cGPU 3.11.0-18-generic #32-Ubuntu SMP Tue Feb 18 21:11:14 UTC 2014 x86_64 x86_64 x86_64 GNU/Linux

nvidia: godzilla@K20cGPU:~$ cat /proc/driver/nvidia/version 
NVRM version: NVIDIA UNIX x86_64 Kernel Module  319.32  Wed Jun 19 15:51:20 PDT 2013
GCC version:  gcc version 4.8.1 (Ubuntu/Linaro 4.8.1-10ubuntu9) 

cuda_device: godzilla@K20cGPU:~$ cat /proc/driver/nvidia/gpus/0/information 
Model: 		 Tesla K20c
IRQ:   		 40

godzilla@K20cGPU:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2013 NVIDIA Corporation
Built on Wed_Jul_17_18:36:13_PDT_2013
Cuda compilation tools, release 5.5, V5.5.0

gcc: godzilla@K20cGPU:~$ gcc --version
gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1

python: In [3]: print sys.version 2.7.6 (default, Feb 26 2014, 00:34:35)   [GCC 4.8.2]

numpy: In [11]: print numpy.version.version   1.8.1rc1

pycuda: In [3]: pycuda.VERSION
Out[3]: (2013, 1, 1)

pytables: In [15]: tables.getPyTablesVersion()  Out[15]: '3.1.0'
hdf: In [8]: tables.getHDF5Version()
Out[8]: '1.8.10-patch1'
"""

def _doItLineNo( cmd, lineNum):
    cmd = shlex.split(cmd)
    ret = subprocess.check_output(cmd).split('\n')
    return ret[lineNum]

def _doIt( cmd):
    cmd = shlex.split(cmd)
    return subprocess.check_output(cmd)

def _get_os():
    _os = _doItLineNo('uname -a', 0).split()
    _os = sjoin( _os[2:5], ':') + ":" + _os[11]
    return _os

def _get_nvrm():
    _nvrm = _doItLineNo('cat /proc/driver/nvidia/version', 0)
    match=re.search(r"\d{3}\.\d{2}", _nvrm)
    return match.group(0)

def _get_cudadev():
    dev = _doItLineNo('cat /proc/driver/nvidia/gpus/0/information ', 0)
    return sjoin(  dev.split()[1:], ' ')

def _get_cudatoolkit():
    ctk = _doItLineNo('nvcc --version', 3)
    match = re.search('V\d+\.\d+\.\d+$', ctk)
    return match.group(0)

def _get_gcc():
    gcc = _doItLineNo('gcc --version', 0)
    return gcc.split()[-1]
    
def _get_pycuda():
    ver = ''
    for sym in CUDAVERSION:
        ver = ver+'%d-'%sym
    return ver[:-1]

def _gitVersionDetector():

    retL = _doIt('git branch -l').split('\n')
    for line in retL:
        if line.startswith('*'): break
    branch = line.split(' ')[1]

    retL = _doIt('git show-ref | grep refs/heads/%s'%branch).split('\n')[0]
    code = retL.split(' ')[0]

    return code, branch

def get_version_info():
    row={}
    row['os'] = _get_os()
    row['nvidia'] = _get_nvrm()
    row['cuda_device'] = _get_cudadev()
    row['cuda_toolkit'] = _get_cudatoolkit()
    row['gcc'] = _get_gcc()
    row['python'] = PYVERSION.split()[0]
    row['numpy'] = numpy_version.version
    row['pycuda'] = _get_pycuda()
    row['pytables'] = getPyTablesVersion()
    row['code_git'] = sjoin( _gitVersionDetector(), ':')
    return row

if __name__ == '__main__':
    
    rowD = get_version_info()
    print rowD

