import os
import ctypes
lualib = ctypes.CDLL(os.path.expanduser("~") + "/torch/install/lib/libluajit.so", mode=ctypes.RTLD_GLOBAL)

import inspect
# We need to enable global symbol visibility for lupa in order to
# support binary module loading in Lua.  If we can enable it here, we
# do it temporarily.

def _try_import_with_global_library_symbols():
    try:
        import DLFCN
        dlopen_flags = DLFCN.RTLD_NOW | DLFCN.RTLD_GLOBAL
    except ImportError:
        import ctypes
        dlopen_flags = ctypes.RTLD_GLOBAL

    import sys
    old_flags = sys.getdlopenflags()
    try:
        sys.setdlopenflags(dlopen_flags)
        import lutorpy._lupa
    finally:
        sys.setdlopenflags(old_flags)

try:
    _try_import_with_global_library_symbols()
except:
    pass

del _try_import_with_global_library_symbols

# the following is all that should stay in the namespace:

from lutorpy._lupa import *

try:
    from lutorpy.version import __version__
except ImportError:
    pass

import lutorpy
luaRuntime = lutorpy._lupa.LuaRuntime()

def LuaRuntime(*args, **kwargs):
    global luaRuntime
    luaRuntime = lutorpy._lupa.LuaRuntime(*args, **kwargs)
    return luaRuntime

globals_ = None
builtins_ = None
warningList = []
def update_globals(verbose = False):
    if globals_ is None:
        return
    lg = luaRuntime.globals()
    for k in lg:
        ks = str(k)
        if ks in builtins_ or globals_.has_key(ks):
            if ks in builtins_ or inspect.ismodule(globals_[ks]):
                if not ks in warningList:
                    warningList.append(ks)
                    if verbose:
                        print('WARNING: variable "'+ ks + '" is already exist in python, use "' + ks + '_" to refer to the lua version')
                globals_[ks + '_'] = lg[ks]
                continue
        globals_[ks] = lg[ks]

def set_globals(g, bi, verbose=True):
    global globals_,builtins_,warningList,require
    warningList = []
    builtins_ = dir(bi)
    globals_ = g
    g['require'] = require
    update_globals(verbose)
    
def eval(cmd):
    ret = luaRuntime.eval(cmd)
    update_globals()
    return ret

def execute(cmd):
    ret = luaRuntime.execute(cmd)
    update_globals()
    return ret

def require(module_name):
    ret = luaRuntime.require(module_name)
    update_globals()
    return ret

def table(*args, **kwargs):
    ret = luaRuntime.table(*args, **kwargs)
    update_globals()
    return ret

def table_from(*args, **kwargs):
    ret = luaRuntime.table_from(*args, **kwargs)
    update_globals()
    return ret

def boostrap_self(obj,func_name):
    '''
        bootstrap a function to add self as the first argument
    '''
    if obj[func_name+'_']:
        return
    func = obj[func_name]
    def func_self(*opt):
        func(obj,*opt)
    obj[func_name+'_'] = func
    obj[func_name] = func_self

bs = boostrap_self


def array2tensor(nparray):
    import numpy as np
    # Byte , Char , Short , Int , Long , Float , and Double
    npType2tensorType = {'int8':'torch.ByteTensor',
                         'int8':'torch.CharTensor',
                         'int16':'torch.ShortTensor',
                         'int32':'torch.IntTensor',
                         'int64':'torch.LongTensor',
                         'float32':'torch.FloatTensor',
                         'float64':'torch.DoubleTensor'
                        }
    luaRuntime.require('torch')
    dtype = str(nparray.dtype)
    if npType2tensorType.has_key(dtype):
        tensorType = npType2tensorType[dtype]
        t = luaRuntime.eval(tensorType+str(nparray.shape).replace(',)',')'))
        ts = t.storage(t)
        t.copyNumpyArray(nparray)
        return t
    else:
        print('Unsupported numpy data type:'+str(nparray.dtype))
    