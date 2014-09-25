from adc_plot_ui import Ui_AdcPlotDialog

from PyQt4.QtCore import *
from PyQt4.QtGui import *


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
import numpy as np
import IPython
import time

import sys

import kid_readout.utils.roach_interface

class AdcPlotDialog(QDialog,Ui_AdcPlotDialog):
    def __init__(self,  qApp, parent=None):
        super(AdcPlotDialog, self).__init__(parent)
        self.__app = qApp
        self.setupUi(self)
        
        self.dpi = 72
        self.fig = Figure((9.1, 5.2), dpi=self.dpi)
#        self.fig = Figure(dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.verticalLayout.insertWidget(0,self.canvas)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.axes = self.fig.add_subplot(221)
        self.axes2 = self.fig.add_subplot(222)
        self.axes3 = self.fig.add_subplot(223)
        self.axes4 = self.fig.add_subplot(224)
        # Use matplotlib event handler
#        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.mpl_toolbar = NavigationToolbar(self.canvas,self.navbar) #self.adc_plot_box)
        
        self.line = None
        self.line2 = None
        self.line3 = None
        self.line4 = None
        
        self.pause_update = False
        
        self.ri = kid_readout.utils.roach_interface.RoachBasebandWide()
        self.ri.set_tone_freqs(np.array([77.915039]),nsamp=2**22)
        
#        self.adc_atten_spin.editingFinished.connect(self.on_adc_atten)
#        self.dac_atten_spin.editingFinished.connect(self.on_dac_atten)
        self.push_apply_atten.clicked.connect(self.apply_atten)
        self.push_tone.clicked.connect(self.onpush_set_tone)
        QTimer.singleShot(1000, self.update_all)
 
    @pyqtSlot()
    def onpush_set_tone(self):
        frq = float(self.line_tone_freq.text())
        self.pause_update = True
        self.ri.set_tone_freqs(np.array([frq]),nsamp=2**20)
        self.pause_update = False
    @pyqtSlot()       
    def apply_atten(self):
        self.on_adc_atten()
        self.on_dac_atten()
        self.ri.set_fft_gain(int(self.spin_fft_gain.value()))
    @pyqtSlot()
    def on_adc_atten(self):
        val = self.adc_atten_spin.value()
        self.ri.set_adc_attenuator(val)
    
    @pyqtSlot()
    def on_dac_atten(self):
        val = self.dac_atten_spin.value()
        self.ri.set_dac_attenuator(val)
            
    def update_all(self):
        tic = time.time()
        if not self.pause_update:
            self.plot_adc()
        self.status_label.setText("%.3f" % (time.time()-tic))
        QTimer.singleShot(1000, self.update_all)
        
    def plot_adc(self):
        x,y = self.ri.get_raw_adc()
        pxx,fr = matplotlib.mlab.psd(x,NFFT=1024,Fs=self.ri.fs*1e6,scale_by_freq = True)
        fr = fr/1e6
        pxx = 10*np.log10(pxx)
        t = np.arange(len(x))/self.ri.fs
        demod = self.check_demod.isChecked()
        d,addr = self.ri.get_data(2,demod=demod)
        print d.shape
        nfft = np.min((1024*32,d.shape[0]/16))
        dpxx,dfr = matplotlib.mlab.psd(d[:,0],NFFT=nfft,Fs=self.ri.fs*1e6/(2.0*self.ri.nfft),scale_by_freq=True)
        dpxx = 10*np.log10(dpxx)
        if self.line:
#            xlim = self.axes.get_xlim()
#            ylim = self.axes.get_ylim()
            self.line.set_xdata(fr)
            self.line.set_ydata(pxx)
            self.line2.set_ydata(x)
            self.line3.set_data(d[:,0].real,d[:,0].imag)
            self.line4.set_ydata(dpxx)
        else:
            self.line, = self.axes.plot(fr,pxx)
            self.line2, = self.axes2.plot(t,x)
            self.line3, = self.axes3.plot(d[:,0].real,d[:,0].imag,'.')
            self.line4, = self.axes4.plot(dfr,dpxx)
            self.axes4.set_xscale('symlog')
            self.axes.set_xlabel('MHz')
            self.axes.set_ylabel('dB/Hz')
            self.axes.grid(True)
            self.axes2.set_xlabel('$\mu$s')
            self.axes2.set_xlim(0,1.0)
            self.axes4.set_xlabel('Hz')
            self.axes4.set_ylabel('dB/Hz')
            self.axes3.set_xlim(-2.**15,2**15)
            self.axes3.set_ylim(-2.**15,2**15)
            self.axes3.hlines([-2**15,2**15],-2**15,2**15)
            self.axes3.vlines([-2**15,2**15],-2**15,2**15)

        self.canvas.draw()
    
        
def main():
    app = QApplication(sys.argv)
    app.quitOnLastWindowClosed = True
    form = AdcPlotDialog(app)
    form.show()
#    form.raise_()
#    app.connect(form, SIGNAL('closeApplication'), app.exit)
    IPython.embed()
#    form.exec_()
    app.exit()
#    sys.exit(app.exec_())        
    
if __name__ == "__main__":
    main()    
