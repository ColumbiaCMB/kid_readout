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
        self.axes = self.fig.add_subplot(211)
        self.axes2 = self.fig.add_subplot(212)
        # Use matplotlib event handler
#        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.mpl_toolbar = NavigationToolbar(self.canvas,self.navbar) #self.adc_plot_box)
        
        self.line = None
        self.line2 = None
        
        self.ri = kid_readout.utils.roach_interface.RoachBaseband()
        
        
#        self.adc_atten_spin.editingFinished.connect(self.on_adc_atten)
#        self.dac_atten_spin.editingFinished.connect(self.on_dac_atten)
        self.push_apply_atten.clicked.connect(self.apply_atten)
        QTimer.singleShot(1000, self.update_all)
 
    @pyqtSlot()       
    def apply_atten(self):
        self.on_adc_atten()
        self.on_dac_atten()
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
        self.plot_adc()
        self.status_label.setText("%.3f" % (time.time()-tic))
        QTimer.singleShot(1000, self.update_all)
        
    def plot_adc(self):
        x,y = self.ri.get_raw_adc()
        pxx,fr = matplotlib.mlab.psd(x,NFFT=1024,Fs=self.ri.fs*1e6,scale_by_freq = True)
        fr = fr/1e6
        pxx = 10*np.log10(pxx)
        t = np.arange(len(x))/self.ri.fs
        if self.line:
#            xlim = self.axes.get_xlim()
#            ylim = self.axes.get_ylim()
            self.line.set_xdata(fr)
            self.line.set_ydata(pxx)
            self.line2.set_ydata(x)
        else:
            self.line, = self.axes.plot(fr,pxx)
            self.line2, = self.axes2.plot(t,x)
            self.axes.set_xlabel('MHz')
            self.axes.set_ylabel('dB/Hz')
            self.axes.grid(True)
            self.axes2.set_xlabel('$\mu$s')
            self.axes2.set_xlim(0,1.0)

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