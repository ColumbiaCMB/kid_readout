import numpy as np
import glob
import os
import kid_readout.utils.plot_nc

all_ncs = (glob.glob('/home/readout/data/*.nc') +
           glob.glob('/home/gjones/data/*.nc') +
           glob.glob('/home/data/*.nc'))

all_ncs.sort()

redo = False

for fname in all_ncs:
    try:
        figdict = kid_readout.utils.plot_nc.plot_sweeps(fname,figsize=(15,15))
        if not figdict:
            continue
        dname,fbase = os.path.split(fname)
        fbase,fext = os.path.splitext(fbase)
        outdirname = os.path.join(dname,fbase)
        if not os.path.exists(outdirname):
            os.mkdir(outdirname)
        elif not redo:
            continue
        for (name,(fig,coarse)) in figdict.items():
            if coarse:
                head = 'coarse'
            else:
                head = 'fine'
            figname = os.path.join(outdirname,('%s_%s.pdf' % (head,name)))
            fig.savefig(figname)
            figname = os.path.join(outdirname,('%s_%s.png' % (head,name)))
            fig.savefig(figname)
    except KeyboardInterrupt:
        break
    except Exception,e:
        print e,fname
        