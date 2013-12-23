"""
easync.py - easier access to netCDF4 files

"""

import netCDF4            
            
class EasyGroup(object):
    def __repr__(self):
        return "EasyNC: %s %s" % (self._filename,self.group.path)
    def __str__(self):
        return self.__repr__()
    def __init__(self,group,filename):
        self._filename = filename
        self.group = group
        self.groups = group.groups
        self.variables = group.variables
        self.dimensions = group.dimensions
        for gname in group.groups.keys():
            if hasattr(self,gname):
                print self,"already has an attribute",gname,"skipping"
                continue
            self.__setattr__(gname,EasyGroup(group.groups[gname],self._filename))
        for vname in group.variables.keys():
            if hasattr(self,vname):
                print self,"already has an attribute",vname,"skipping"
                continue
            self.__setattr__(vname,group.variables[vname])
        for dname in group.dimensions.keys():
            dimname = "dim_" + dname
            if hasattr(self,dimname):
                print self,"already has an attribute",dimname,"skipping"
                continue
            self.__setattr__(dimname,group.dimensions[dname])
def EasyNetCDF4(*args,**kwargs):
    nc = netCDF4.Dataset(*args,**kwargs)
    if len(args) > 0:
        fn = args[0]
    else:
        fn = kwargs['filename']
    enc =  EasyGroup(nc,fn)
    enc.close = nc.close
    enc.sync = nc.sync
    return enc