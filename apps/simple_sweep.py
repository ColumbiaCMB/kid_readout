from basic_sweep_ui import Ui_SweepDialog

from PyQt4.QtCore import *
from PyQt4.QtGui import *


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import IPython
import time
import bisect

import threading

import sys

import kid_readout.utils.roach_interface
import kid_readout.utils.sweeps
from kid_readout.utils.data_block import SweepData
from kid_readout.utils import data_file
#from kid_readout.utils.PeakFind01 import peakdetect


class SweepDialog(QDialog,Ui_SweepDialog):
    def __init__(self,  qApp, parent=None):
        super(SweepDialog, self).__init__(parent)
        self.__app = qApp
        self.setupUi(self)
        self.dpi = 72
        self.fig = Figure((9.1, 5.2), dpi=self.dpi)
#        self.fig = Figure(dpi=self.dpi)
        self.plot_layout = QVBoxLayout(self.plot_group_box)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.plot_group_box)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.plot_layout.addWidget(self.canvas)
        self.axes = self.fig.add_subplot(211)
        self.axes.set_xlabel('MHz')
        self.axes.set_ylabel('dB')
        self.axes.grid(True)
        self.axes2 = self.fig.add_subplot(212)
        # Use matplotlib event handler
        #self.canvas.mpl_connect('pick_event', self.onclick_plot)
        self.canvas.mpl_connect('button_release_event', self.onclick_plot)
        self.mpl_toolbar = NavigationToolbar(self.canvas,self.plot_group_box) #self.adc_plot_box)
        self.plot_layout.addWidget(self.mpl_toolbar)
        
        self.cal_freq=np.arange(256)
        self.cal_mag = np.ones((256,))
        self.cal_phase = np.zeros((256,))
        
        try:
            cdata = np.load('lbcal1.npz')
            self.cal_freq = cdata['freq']
            self.cal_mag = cdata['mag']
            self.cal_phase = cdata['phase']
        except Exception, e:
            print "cal data could not be loaded",e
            
        self.line = None
        self.phline = None
        self.line2 = None
        self.phline2 = None
        self.peakline = None
        self.psd_text = None
        self.selection_line = None
        self.sweep_data = None
        self.fine_sweep_data = None
        
        self.selected_sweep = 'coarse'
        self.selected_idx = 0
        self.progress_value = 0
        self.total_subsweeps = 1
        self.current_subsweep = 0
        
        self.line_dac_gain.setText("-2.0")
        
        self.ri = kid_readout.utils.roach_interface.RoachBaseband()
        #self.ri.set_adc_attenuator(31)
        self.ri.set_dac_attenuator(36)
        
        self.abort_requested = False
        self.sweep_thread = None
        
        self.reslist = np.array([92.94,
                         96.31,
                         101.546,
                         117.195,
                         121.35,
                         130.585,
                         133.436,
                         148.238,
                         148.696,
                         148.867,
                         149.202,
                         159.572,
                         167.97,
                         172.93,
                         176.645,
                         178.764])
        self.setup_freq_table()
        
        self.push_abort.clicked.connect(self.onclick_abort)
        self.push_start_sweep.clicked.connect(self.onclick_start_sweep)
        self.push_start_fine_sweep.clicked.connect(self.onclick_start_fine_sweep)
        self.push_save.clicked.connect(self.onclick_save)
        self.line_npoints.textEdited.connect(self.recalc_spacing)
        self.line_span_hz.textEdited.connect(self.recalc_spacing)
        self.tableview_freqs.itemChanged.connect(self.freq_table_item_changed)
        self.spin_subsweeps.valueChanged.connect(self.onspin_subsweeps_changed)
        self.push_add_resonator.clicked.connect(self.onclick_add_resonator)
        self.push_clear_all.clicked.connect(self.onclick_clear_all)
        self.check_use_cal.stateChanged.connect(self.oncheck_use_cal)
        self.push_save_res.clicked.connect(self.onclick_save_res)
        self.push_load_res.clicked.connect(self.onclick_load_res)
        
        self.logfile = None
        self.fresh = False
        self.fine_sweep_data = None
        self.recalc_spacing('')
        
        self.onspin_subsweeps_changed(1)
        QTimer.singleShot(1000, self.update_plot)
        
    def onclick_plot(self,event):
        print event.xdata,event.ydata,event.inaxes
        if event.inaxes == self.axes:
            if event.button != 1 and self.fine_sweep_data is not None:
                sweep_data = self.fine_sweep_data
                self.selected_sweep = 'fine'
            else:
                sweep_data = self.sweep_data
                self.selected_sweep = 'coarse'
            idx = (np.abs(sweep_data.freqs - event.xdata)).argmin()
            self.selected_idx = idx
            self.axes2.cla()
            NFFT = sweep_data.blocks[idx].data.shape[0]/2
            pxx,fr = plt.mlab.psd(sweep_data.blocks[idx].data,Fs=256e6/2**14,NFFT=NFFT,detrend=plt.mlab.detrend_mean)
            self.axes2.semilogx(fr[fr>0],10*np.log10(pxx[fr>0]))
            self.axes2.semilogx(-fr[fr<0],10*np.log10(pxx[fr<0]))
            self.axes2.semilogx(fr[fr>0][0],10*np.log10(np.abs(sweep_data.blocks[idx].data.mean())**2*NFFT/(256e6/2**14)), 'o', mew=2)
            blk = sweep_data.blocks[idx]
            freq = blk.fs*blk.tone/float(blk.nsamp)
            self.axes2.text(0.95,0.95,('%.6f MHz' % freq), ha='right', va='top',
                            transform = self.axes2.transAxes)
            self.axes2.set_xlim(10,fr[-1])
            self.axes2.set_ylim(-140,-20)
            self.axes2.grid(True)
            self.axes2.set_ylabel('dB/Hz')
            self.axes2.set_xlabel('Hz')
            #self.axes2.semilogx(fr[fr>0][0],10*np.log10(pxx[fr==0]), 'o', mew=2)
            #self.canvas.draw()
            self.fresh = True
            print idx
        
    def update_plot(self):
        if self.fresh and (self.sweep_data is not None or self.fine_sweep_data is not None):
            x = self.sweep_data.freqs
            y = self.sweep_data.data
            if self.check_use_cal.isChecked():
                cal = np.interp(x,self.cal_freq,self.cal_mag)
                cal *= 10**(-(self.ri.adc_atten+self.ri.dac_atten-31)/20.0)
            else:
                cal = 1.0
            ph = self.sweep_data.data[:]
            if len(x) >0 and len(x) == len(y) and len(x) == len(ph):
                y = 20*np.log10(np.abs(y/cal))
                if len(self.reslist):
                    resy = np.interp(self.reslist, x, y)
                else:
                    resy = np.zeros((0,))
                ph = np.angle(ph*np.exp(-1j*x*398.15))
                if self.line:
                    self.line.set_xdata(x)
                    self.line.set_ydata(y)
#                    self.phline.set_xdata(x)
#                    self.phline.set_ydata(ph)
                    self.peakline.set_data(self.reslist,resy)
                    
                else:
                    self.line, = self.axes.plot(x,y,'b.-',alpha=0.5)
#                    self.phline, = self.axes.plot(x,ph,'g',alpha=0)
                    self.peakline, = self.axes.plot(self.reslist,resy,'ro')
                    
                if self.selected_sweep == 'coarse':
                    if self.selected_idx >= len(x):
                        self.selected_idx = 0
                    if self.selection_line:
                        self.selection_line.set_data([x[self.selected_idx]],[y[self.selected_idx]])
                    else:
                        self.selection_line, = self.axes.plot([x[self.selected_idx]],[y[self.selected_idx]],'mx',mew=2,markersize=20)

                if self.fine_sweep_data is not None:
                    x = self.fine_sweep_data.freqs
                    y = self.fine_sweep_data.data
                    ph = self.fine_sweep_data.data[:]
                    if len(x) == len(y) and len(x) == len(ph):
                        if self.check_use_cal.isChecked():
                            cal = np.interp(x,self.cal_freq,self.cal_mag)
                            cal *= 10**(-(self.ri.adc_atten+self.ri.dac_atten-31)/20.0)
                        else:
                            cal = 1.0
                        y = 20*np.log10(np.abs(y/cal))
                        ph = np.angle(ph*np.exp(-1j*x*398.15))
                        if self.line2:
                            self.line2.set_xdata(x)
                            self.line2.set_ydata(y)
#                            self.phline2.set_xdata(x)
#                            self.phline2.set_ydata(ph)
                        else:
                            self.line2, = self.axes.plot(x,y,'r.',alpha=0.5)
#                            self.phline2, = self.axes.plot(x,ph,'k.',alpha=0)
                    if self.selected_sweep == 'fine':
                        if self.selected_idx >= len(x):
                            self.selected_idx = 0

                        if self.selection_line:
                            self.selection_line.set_data([x[self.selected_idx]],[y[self.selected_idx]])
                        else:
                            self.selection_line, = self.axes.plot([x[self.selected_idx]],[y[self.selected_idx]],'mx',mew=2,markersize=20)

                self.canvas.draw()
            self.fresh = False
        
        self.line_dac_atten.setText(str(self.ri.dac_atten))
        self.progress_sweep.setValue(int(self.progress_value*100))
        QTimer.singleShot(1000, self.update_plot)
        
    @pyqtSlot(int)
    def oncheck_use_cal(self,val):
        self.fresh = True
                
    @pyqtSlot(int)
    def onspin_subsweeps_changed(self, val):
        step = 0.0625*2**self.combo_step_size.currentIndex()
        substep = step/float(val)
        nsamp = np.ceil(np.log2(self.ri.fs/substep))
        if nsamp < 18:
            nsamp = 18
        self.label_coarse_info.setText("Spacing: %.3f kHz using 2**%d samples" % (substep*1000,nsamp))
    @pyqtSlot()
    def onclick_add_resonator(self):
        if self.selected_idx is not None:
            if self.selected_sweep == 'coarse':
                freq = self.sweep_data.freqs[self.selected_idx]
            else:
                freq = self.fine_sweep_data.freqs[self.selected_idx]
        reslist = self.reslist.tolist()
        bisect.insort(reslist,freq)
        self.reslist = np.array(reslist)
        self.refresh_freq_table()
        
    @pyqtSlot()
    def onclick_save_res(self):
        fname = str(QFileDialog.getSaveFileName(self, "Save resonators as:",".", "Numpy (*.npy)"))
        np.save(fname,self.reslist)
    
    @pyqtSlot()
    def onclick_load_res(self):
        fname = str(QFileDialog.getOpenFileName(self, "Load resonators from:",".", "Numpy (*.npy)"))
        if fname:
            reslist = np.load(fname)
            self.reslist = reslist
            self.refresh_freq_table()
            
    @pyqtSlot()
    def onclick_clear_all(self):
        self.reslist = np.array([])
        self.refresh_freq_table()
    @pyqtSlot()
    def onclick_save(self):
        if self.logfile:
            self.logfile.close()
            self.logfile = None
            self.push_save.setText("Start Logging")
            self.line_filename.setText('')
        else:
            self.logfile = data_file.DataFile()  
            self.line_filename.setText(self.logfile.filename)
            self.push_save.setText("Close Log File")
    @pyqtSlot()
    def onclick_abort(self):
        self.abort_requested = True
        
    @pyqtSlot(str)
    def recalc_spacing(self,txt):
        msg = None
        span = None
        npoint = None
        try:
            span = float(self.line_span_hz.text())
            if span <= 0:
                raise Exception()
        except:
            msg = "span invalid"
        try:
            npoint = int(self.line_npoints.text())
            if npoint <=0:
                raise Exception()
        except:
            msg = "invalid number of points"
        if msg:
            self.label_spacing.setText(msg)
        else:
            spacing = span/npoint
            samps = np.ceil(np.log2(self.ri.fs*1e6/spacing))
            self.label_spacing.setText("Spacing: %.3f Hz requires 2**%d samples" % (spacing,samps))
    def sweep_callback(self,block):
        self.sweep_data.add_block(block)
        self.fresh = True
        self.progress_value = (block.progress + self.current_subsweep)/float(self.total_subsweeps)
#        print "currently have freqs", self.sweep_data.freqs
        return self.abort_requested
    
    def fine_sweep_callback(self,block):
        self.fine_sweep_data.add_block(block)
        self.fresh = True
        self.progress_value = (block.progress + self.current_subsweep)/float(self.total_subsweeps)
        return self.abort_requested
    
    @pyqtSlot()
    def onclick_start_sweep(self):
        if self.sweep_thread:
            if self.sweep_thread.is_alive():
                print "sweep already running"
                return
        self.sweep_thread = threading.Thread(target=self.do_sweep)
        self.sweep_thread.daemon = True
        self.sweep_thread.start()
        
    @pyqtSlot()
    def onclick_start_fine_sweep(self):
        if np.mod(self.reslist.shape[0],4) != 0:
            print "Number of resonators must be divisible by 4! Add some dummy resonators."
        if self.sweep_thread:
            if self.sweep_thread.is_alive():
                print "sweep already running"
                return
        self.sweep_thread = threading.Thread(target=self.do_fine_sweep)
        self.sweep_thread.daemon = True
        self.sweep_thread.start()
        
    def do_sweep(self):
        self.abort_requested = False
        self.sweep_data = SweepData(sweep_id=1)
        start = self.spin_start_freq.value()
        stop = self.spin_stop_freq.value()
        step = 0.0625*2**self.combo_step_size.currentIndex()
        nsubstep = self.spin_subsweeps.value()
        substepspace = step/nsubstep
        nsamp = np.ceil(np.log2(self.ri.fs/substepspace))
        if nsamp < 18:
            nsamp = 18
        self.total_subsweeps = nsubstep
        base_freqs = np.arange(start,stop+1e-3,step)
        ntones = base_freqs.shape[0]
        ntones_corr = kid_readout.utils.roach_interface.ntone_power_correction(ntones)
        try:
            dac_gain = float(self.line_dac_gain.text())
        except:
            print "Could not parse DAC gain! Aborting sweep!"
            return
        desired_level = self.spin_dbm_per_tone.value()
        total_level = desired_level + ntones_corr
        atten_val = -(total_level - dac_gain)  #atten should be positive
        print "Requested %.2f dBm per tone" % desired_level
        print "%d tones requires a correction of %.1f dB" % (ntones,ntones_corr)
        print "requiring %.1f dB of attenuation" % atten_val
        
        if atten_val < 0:
            print "Warning! requested attenuation is less than 0, can't reach desired level. Setting to 0 dB"
            atten_val = 0
        if atten_val > 63:
            print "Warning! requested attenuation is greater than 63, can't reach desired level. Setting to 63 dB"
            atten_val = 63
        self.ri.set_dac_attenuator(atten_val)
        for k in range(nsubstep):
            self.current_subsweep = k
            print "subsweep",k,"of",nsubstep
            if self.logfile:
                self.logfile.log_hw_state(self.ri)
            kid_readout.utils.sweeps.coarse_sweep(self.ri, freqs = base_freqs + k*substepspace, 
                                                  nsamp = 2**nsamp, nchan_per_step=8, callback=self.sweep_callback, sweep_id=1)
            if self.logfile:
                self.logfile.log_adc_snap(self.ri)
            if self.abort_requested:
                break
        if self.logfile:
            name = self.logfile.add_sweep(self.sweep_data)
            self.label_status.setText("saved %s" % name)
            
        #self.find_resonances()
        self.refresh_freq_table()
        self.abort_requested = False
        
    def do_fine_sweep(self):
        self.abort_requested = False
        self.fine_sweep_data = SweepData(sweep_id=2)
        try:
            width = float(self.line_span_hz.text())/1e6
            npts = int(self.line_npoints.text())
        except:
            print "npoint or span is invalid"
            return
        spacing = width/npts
        samps = np.ceil(np.log2(self.ri.fs/spacing))
        print samps
        flist = self.reslist
        offsets = np.linspace(-width/2,width/2,npts)
        self.total_subsweeps = len(offsets)
        
        ntones = flist.shape[0]
        ntones_corr = kid_readout.utils.roach_interface.ntone_power_correction(ntones)
        try:
            dac_gain = float(self.line_dac_gain.text())
        except:
            print "Could not parse DAC gain! Aborting sweep!"
            return
        desired_level = self.spin_dbm_per_tone.value()
        total_level = desired_level + ntones_corr
        atten_val = -(total_level - dac_gain)  #atten should be positive
        print "Requested %.2f dBm per tone" % desired_level
        print "%d tones requires a correction of %.1f dB" % (ntones,ntones_corr)
        print "requiring %.1f dB of attenuation" % atten_val
        
        if atten_val < 0:
            print "Warning! requested attenuation is less than 0, can't reach desired level. Setting to 0 dB"
            atten_val = 0
        if atten_val > 63:
            print "Warning! requested attenuation is greater than 63, can't reach desired level. Setting to 63 dB"
            atten_val = 63
        self.ri.set_dac_attenuator(atten_val)
            
        for k,offs in enumerate(offsets):
            self.current_subsweep = k
            if self.logfile:
                self.logfile.log_hw_state(self.ri)            
            kid_readout.utils.sweeps.coarse_sweep(self.ri, freqs = flist+offs, 
                                              nsamp = 2**samps, nchan_per_step=8, callback=self.fine_sweep_callback, sweep_id=2)
            if self.abort_requested:
                break
            if self.logfile:
                self.logfile.log_adc_snap(self.ri)
        if self.logfile:
            name = self.logfile.add_sweep(self.fine_sweep_data)
            self.label_status.setText("saved %s" % name)
        self.abort_requested = False
        
    def find_resonances(self):
        x = self.sweep_data.freqs
        y = np.abs(self.sweep_data.data)
        mx,mn = peakdetect(y,x,lookahead=20)
        res = np.array(mn)
        if len(res) == 0:
            self.reslist=np.array([])
            return
        self.reslist = np.array(mn)[:,0]
        self.fresh = True
        
    def setup_freq_table(self):
        self.tableview_freqs.clear()
        self.tableview_freqs.setSortingEnabled(False)
        self.tableview_freqs.setAlternatingRowColors(True)
        self.tableview_freqs.setSelectionBehavior(QTableWidget.SelectRows)

        self.tableview_freqs.setColumnCount(1)
#        self.tableview_freqs.setColumnWidth(0, 80)

        self.tableview_freqs.setRowCount(64)#self.reslist.shape[0])

        headers = ['f0']
        self.tableview_freqs.setHorizontalHeaderLabels(headers)
        self.refresh_freq_table()

    def refresh_freq_table(self):
        self.tableview_freqs.clear()
        self.tableview_freqs.blockSignals(True)
        for row,f0 in enumerate(self.reslist):
            # Current frequency            
            item = QTableWidgetItem(QString.number(float(f0), 'f', 6))
            item.setTextAlignment(Qt.AlignRight)
            self.tableview_freqs.setItem(row, 0, item)
        for row in range(self.reslist.shape[0],64):
            item = QTableWidgetItem('')
            item.setTextAlignment(Qt.AlignRight)
            self.tableview_freqs.setItem(row, 0, item)
        self.tableview_freqs.blockSignals(False)
        self.fresh = True
        
    @pyqtSlot(QTableWidgetItem)
    def freq_table_item_changed(self,item):
        flist = []
        for row in range(self.tableview_freqs.rowCount()):
            try:
                val = float(self.tableview_freqs.item(row,0).text())
                flist.append(val)
            except ValueError:
                pass
        flist.sort()
        self.reslist = np.array(flist)
        self.refresh_freq_table()

def main():
    app = QApplication(sys.argv)
    app.quitOnLastWindowClosed = True
    form = SweepDialog(app)
    form.setAttribute(Qt.WA_QuitOnClose)
    form.setAttribute(Qt.WA_DeleteOnClose)
    try:
        from kid_readout.utils.borph_utils import check_output
        cmd = ('git log -1 --pretty=format:"%%ci" %s' % __file__)
#        print cmd
        dcode = check_output(cmd, shell=True)
    except:
        dcode = ''
    form.setWindowTitle("KIDleidoscope %s" % dcode)
    
    form.show()
#    form.raise_()
#    app.connect(form, SIGNAL('closeApplication'), sys.exit)#app.exit)
#    print "starting ipython"
    IPython.embed()
#    form.exec_()
#    print "after ipython embed"
    if form.logfile:
        form.logfile.close()
    app.exit()
#    sys.exit()
#    app.exec_()
#    print "after app exec"
    
if __name__ == "__main__":
    main()    
