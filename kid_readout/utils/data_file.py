import netCDF4
import time
import os
import numpy as np
from kid_readout.utils.valon import check_output

class DataFile():
    def __init__(self,base_dir='/home/data'):
        base_dir = os.path.expanduser(base_dir)
        if not os.path.exists(base_dir):
            try:
                os.mkdir(base_dir)
            except Exception, e:
                raise Exception("Tried to make directory %s for data file but failed. Error was %s" % (base_dir,str(e)))
        fn = time.strftime('%Y-%m-%d_%H%M%S.nc')
        fn = os.path.join(base_dir,fn)
        self.filename = fn
        self.nc = netCDF4.Dataset(fn,mode='w')
        try:
            dname = os.path.split(__file__)[0]
            gitinfo = check_output(("cd %s; git log -1" % dname),shell=True)
        except:
            gitinfo = ''
        self.nc.gitinfo = gitinfo
        self.sweeps = self.nc.createGroup('sweeps')
        self.timestreams = self.nc.createGroup('timestreams')
        self.cryo = self.nc.createGroup('cryo')
        self.hw_state = self.nc.createGroup('hw_state')
        self.hw_state.createDimension('time',None)
        self.hw_epoch = self.hw_state.createVariable('epoch',np.float64,dimensions=('time',))
        self.hw_adc_atten = self.hw_state.createVariable('adc_atten',np.float32,dimensions=('time',))
        self.hw_dac_atten = self.hw_state.createVariable('dac_atten',np.float32,dimensions=('time',))
        self.hw_ntones = self.hw_state.createVariable('ntones',np.int32,dimensions=('time',))
        
        self.adc_snaps = self.nc.createGroup('adc_snaps')
        self.adc_snaps.createDimension('epoch', None)
        self.adc_snaps.createDimension('adc', 2)
        self.adc_snaps.createDimension('sample', 2**12)
        self.adc_snaps_epoch = self.adc_snaps.createVariable('epoch',np.float64,dimensions=('epoch',))
        self.adc_snaps_data = self.adc_snaps.createVariable('data',np.float32,dimensions=('epoch','adc','sample'))
        
        self.c128 = np.dtype([('real','f8'),('imag','f8')])
        self.cdf128 = self.nc.createCompoundType(self.c128, 'complex128')
        self.c64 = np.dtype([('real','f4'),('imag','f4')])
        self.cdf64 = self.nc.createCompoundType(self.c64, 'complex64')
        
    def close(self):
        self.nc.close()        
    def log_hw_state(self,ri):
        idx = len(self.hw_state.dimensions['time'])
        t0 = time.time()
        self.hw_epoch[idx] = t0
        self.hw_adc_atten[idx] = ri.adc_atten
        self.hw_dac_atten[idx] = ri.dac_atten
        self.hw_ntones[idx] = ri.tone_bins.shape[0]
        
    def log_adc_snap(self,ri):
        t0 = time.time()
        idx = len(self.adc_snaps.dimensions['epoch'])
        x,y = ri.get_raw_adc()
        self.adc_snaps_epoch[idx] = t0
        self.adc_snaps_data[idx,0,:] = x
        self.adc_snaps_data[idx,1,:] = y

    def add_sweep(self, sweep_data):
        name = time.strftime('sweep_%Y%m%d%H%M%S')
        swg = self.sweeps.createGroup(name)
        swg.createDimension('frequency',None)
        freq = swg.createVariable('frequency',np.float64,('frequency',))
        s21 = swg.createVariable('s21',self.cdf128,('frequency',))
        index = swg.createVariable('index',np.int32,('frequency',))
        freq[:] = sweep_data.freqs
        s21[:] = sweep_data.data.astype('complex128').view(self.c128)
        index[:] = sweep_data.sweep_indexes
        
        dbg = swg.createGroup('datablocks')
        dbg.createDimension('epoch',None)
        dbg.createDimension('sample',sweep_data.blocks[0].data.shape[0])
        t0 = dbg.createVariable('epoch',np.float64,('epoch',))
        tone = dbg.createVariable('tone',np.int32,('epoch',))
        nsamp = dbg.createVariable('nsamp',np.int32,('epoch',))
        fftbin = dbg.createVariable('fftbin',np.int32,('epoch',))
        nfft = dbg.createVariable('nfft',np.int32,('epoch',))
        dt = dbg.createVariable('dt',np.float64,('epoch',))
        fs = dbg.createVariable('fs',np.float64,('epoch',))
        wavenorm = dbg.createVariable('wavenorm',np.float64,('epoch'))
        data = dbg.createVariable('data',self.cdf128,('epoch','sample'))
        sweep_index = dbg.createVariable('sweep_index',np.int32,('epoch'))
        
        blocks = sweep_data.blocks
        blen = blocks[0].data.shape[0]
        blocklist = []
        for blk in blocks:
            if blk.data.shape[0] >= blen:
                blocklist.append(blk.data[None,:blen])
            else:
                newblk = np.zeros((1,blen),dtype=blk.data.dtype)
                newblk[0,:blk.data.shape[0]] = blk.data[:]
                blocklist.append(newblk)
        data[:] = np.concatenate(blocklist,axis=0).astype('complex128').view(self.c128)
        fs[:] = np.array([x.fs for x in blocks])
        t0[:] = np.array([x.t0 for x in blocks])
        tone[:] = np.array([x.tone for x in blocks])
        nfft[:] = np.array([x.nfft for x in blocks])
        wavenorm[:] = np.array([x.wavenorm for x in blocks])
        nsamp[:] = np.array([x.nsamp for x in blocks])
        dt[:] = np.array([x.dt for x in blocks])
        fftbin[:] = np.array([x.fftbin for x in blocks])
        sweep_index[:] = np.array([x.sweep_index for x in blocks])
        
        return name
        
    def add_block_to_timestream(self, block, tsg = None):
        if tsg is None:
            name = time.strftime('timestream_%Y%m%d%H%M%S')
            tsg = self.timestreams.createGroup(name)
            tsg.createDimension('epoch',None)
            tsg.createDimension('sample',block.data.shape[0])
            t0 = tsg.createVariable('epoch',np.float64,('epoch',))
            tone = tsg.createVariable('tone',np.int32,('epoch',))
            nsamp = tsg.createVariable('nsamp',np.int32,('epoch',))
            fftbin = tsg.createVariable('fftbin',np.int32,('epoch',))
            nfft = tsg.createVariable('nfft',np.int32,('epoch',))
            dt = tsg.createVariable('dt',np.float64,('epoch',))
            fs = tsg.createVariable('fs',np.float64,('epoch',))
            wavenorm = tsg.createVariable('wavenorm',np.float64,('epoch'))
            data = tsg.createVariable('data',self.cdf128,('epoch','sample'))
        else:
            t0 = tsg.variables['epoch']
            tone = tsg.variables['tone']
            nsamp = tsg.variables['nsamp']
            fftbin = tsg.variables['fftbin']
            nfft = tsg.variables['nfft']
            dt = tsg.variables['dt']
            fs = tsg.variables['fs']
            wavenorm = tsg.variables['wavenorm']
            data = tsg.variables['data']
        idx = len(tsg.dimensions['epoch'])
        data[idx] = block.data.astype('complex128').view(self.c128)
        t0[idx] = block.t0
        fs[idx] = block.fs
        tone[idx] = block.tone
        nsamp[idx] = block.nsamp
        fftbin[idx] = block.fftbin
        nfft[idx] = block.nfft
        wavenorm[idx] = block.wavenorm
        dt[idx] = block.dt
        return tsg
    
    def add_cryo_data(self,cryod):
        if not self.cryo.variables:
            self.cryo.createDimension('epoch',None)
            self.cryo.createDimension('input', 4)
            for name,val in cryod.items():
                if type(val) is list:
                    self.cryo.createVariable(name,np.float64,('epoch','input'))
                else:
                    self.cryo.createVariable(name,np.float64,('epoch',))
        idx = len(self.cryo.dimensions['epoch'])
        for name,val in cryod.items():
            if type(val) is list:
                self.cryo.variables[name][idx] = val[:]
            else:
                self.cryo.variables[name][idx] = val      


        
        
