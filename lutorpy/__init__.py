import os
import ctypes
from ctypes.util import find_library
import os
import inspect
from sys import platform as _platform
import builtins

try:
    os.system(os.path.expanduser("~") + "/torch/install/bin/torch-activate")
except:
    pass

if _platform == "linux" or _platform == "linux2":
    lualib = ctypes.CDLL(os.path.expanduser("~") + "/torch/install/lib/libluajit.so", mode=ctypes.RTLD_GLOBAL)
    THlib = ctypes.CDLL(os.path.expanduser("~") + "/torch/install/lib/libTH.so", mode=ctypes.RTLD_GLOBAL)
    luaTlib = ctypes.CDLL(os.path.expanduser("~") + "/torch/install/lib/libluaT.so", mode=ctypes.RTLD_GLOBAL)
elif _platform == "darwin":
    lualib = ctypes.CDLL(find_library('luajit'), mode=ctypes.RTLD_GLOBAL)
    THlib = ctypes.CDLL(find_library('TH'), mode=ctypes.RTLD_GLOBAL)
    luaTlib = ctypes.CDLL(find_library('luaT'), mode=ctypes.RTLD_GLOBAL)
elif _platform == "win32":
    lualib = ctypes.CDLL(find_library('luajit'), mode=ctypes.RTLD_GLOBAL)
    THlib = ctypes.CDLL(find_library('TH'), mode=ctypes.RTLD_GLOBAL)
    luaTlib = ctypes.CDLL(find_library('luaT'), mode=ctypes.RTLD_GLOBAL)

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

def LuaRuntime(*args, **kwargs):
    global luaRuntime
    if not kwargs.has_key('zero_based_index'):
        kwargs['zero_based_index']=True
    luaRuntime = lutorpy._lupa.LuaRuntime(*args, **kwargs)
    return luaRuntime

LuaRuntime()

builtins_ = dir(builtins)
warningList = []
def update_globals(globals_, verbose = False):
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
                        print('WARNING: please use "' + ks + '_" to refer to the lua object "' + ks +'"')
                globals_[ks + '_'] = lg[ks]
                continue
        globals_[ks] = lg[ks]
        
    global require
    globals_['require'] = require
        
def require(module_name):
    ret = luaRuntime.require(module_name)
    stack = inspect.stack()
    try:
        globals_ = stack[1][0].f_globals
        update_globals(globals_, verbose=False)
    finally:
        del stack
    return ret
    
def eval(cmd):
    ret = luaRuntime.eval(cmd)
    stack = inspect.stack()
    try:
        globals_ = stack[1][0].f_globals
        update_globals(globals_, verbose=False)
    finally:
        del stack
    return ret

def execute(cmd):
    ret = luaRuntime.execute(cmd)
    stack = inspect.stack()
    try:
        globals_ = stack[1][0].f_globals
        update_globals(globals_, verbose=False)
    finally:
        del stack
    return ret

def table(*args, **kwargs):
    ret = luaRuntime.table(*args, **kwargs)
    stack = inspect.stack()
    try:
        globals_ = stack[1][0].f_globals
        update_globals(globals_, verbose=False)
    finally:
        del stack
    return ret

def table_from(*args, **kwargs):
    ret = luaRuntime.table_from(*args, **kwargs)
    stack = inspect.stack()
    try:
        globals_ = stack[1][0].f_globals
        update_globals(globals_, verbose=False)
    finally:
        del stack
    return ret


stack = inspect.stack()
try:
    globals_ = stack[1][0].f_globals
    update_globals(globals_, verbose=False)
finally:
    del stack
