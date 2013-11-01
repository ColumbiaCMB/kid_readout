# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/adc_plot.ui'
#
# Created: Fri Nov  1 13:16:16 2013
#      by: PyQt4 UI code generator 4.8.5
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_AdcPlotDialog(object):
    def setupUi(self, AdcPlotDialog):
        AdcPlotDialog.setObjectName(_fromUtf8("AdcPlotDialog"))
        AdcPlotDialog.resize(1024, 760)
        AdcPlotDialog.setWindowTitle(QtGui.QApplication.translate("AdcPlotDialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        AdcPlotDialog.setSizeGripEnabled(True)
        self.verticalLayout = QtGui.QVBoxLayout(AdcPlotDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.navbar = QtGui.QFrame(AdcPlotDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.navbar.sizePolicy().hasHeightForWidth())
        self.navbar.setSizePolicy(sizePolicy)
        self.navbar.setMinimumSize(QtCore.QSize(0, 35))
        self.navbar.setFrameShape(QtGui.QFrame.Box)
        self.navbar.setObjectName(_fromUtf8("navbar"))
        self.verticalLayout.addWidget(self.navbar)
        self.status_label = QtGui.QLabel(AdcPlotDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.status_label.sizePolicy().hasHeightForWidth())
        self.status_label.setSizePolicy(sizePolicy)
        self.status_label.setMinimumSize(QtCore.QSize(0, 10))
        self.status_label.setText(QtGui.QApplication.translate("AdcPlotDialog", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))
        self.status_label.setObjectName(_fromUtf8("status_label"))
        self.verticalLayout.addWidget(self.status_label)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(AdcPlotDialog)
        self.label.setText(QtGui.QApplication.translate("AdcPlotDialog", "ADC Atten:", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.adc_atten_spin = QtGui.QDoubleSpinBox(AdcPlotDialog)
        self.adc_atten_spin.setSuffix(QtGui.QApplication.translate("AdcPlotDialog", " dB", None, QtGui.QApplication.UnicodeUTF8))
        self.adc_atten_spin.setDecimals(1)
        self.adc_atten_spin.setMaximum(31.5)
        self.adc_atten_spin.setSingleStep(0.5)
        self.adc_atten_spin.setProperty("value", 20.0)
        self.adc_atten_spin.setObjectName(_fromUtf8("adc_atten_spin"))
        self.horizontalLayout.addWidget(self.adc_atten_spin)
        self.horizontalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtGui.QLabel(AdcPlotDialog)
        self.label_2.setText(QtGui.QApplication.translate("AdcPlotDialog", "DAC Atten:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.dac_atten_spin = QtGui.QDoubleSpinBox(AdcPlotDialog)
        self.dac_atten_spin.setSuffix(QtGui.QApplication.translate("AdcPlotDialog", " dB", None, QtGui.QApplication.UnicodeUTF8))
        self.dac_atten_spin.setDecimals(1)
        self.dac_atten_spin.setMaximum(31.5)
        self.dac_atten_spin.setSingleStep(0.5)
        self.dac_atten_spin.setProperty("value", 20.0)
        self.dac_atten_spin.setObjectName(_fromUtf8("dac_atten_spin"))
        self.horizontalLayout_2.addWidget(self.dac_atten_spin)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        self.label_3 = QtGui.QLabel(AdcPlotDialog)
        self.label_3.setText(QtGui.QApplication.translate("AdcPlotDialog", "FFT gain", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_3.addWidget(self.label_3)
        self.spin_fft_gain = QtGui.QSpinBox(AdcPlotDialog)
        self.spin_fft_gain.setMaximum(16)
        self.spin_fft_gain.setProperty("value", 1)
        self.spin_fft_gain.setObjectName(_fromUtf8("spin_fft_gain"))
        self.horizontalLayout_3.addWidget(self.spin_fft_gain)
        self.push_apply_atten = QtGui.QPushButton(AdcPlotDialog)
        self.push_apply_atten.setText(QtGui.QApplication.translate("AdcPlotDialog", "Apply", None, QtGui.QApplication.UnicodeUTF8))
        self.push_apply_atten.setObjectName(_fromUtf8("push_apply_atten"))
        self.horizontalLayout_3.addWidget(self.push_apply_atten)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.push_tone = QtGui.QPushButton(AdcPlotDialog)
        self.push_tone.setText(QtGui.QApplication.translate("AdcPlotDialog", "set Tone:", None, QtGui.QApplication.UnicodeUTF8))
        self.push_tone.setObjectName(_fromUtf8("push_tone"))
        self.horizontalLayout_3.addWidget(self.push_tone)
        self.line_tone_freq = QtGui.QLineEdit(AdcPlotDialog)
        self.line_tone_freq.setObjectName(_fromUtf8("line_tone_freq"))
        self.horizontalLayout_3.addWidget(self.line_tone_freq)
        self.checkBox = QtGui.QCheckBox(AdcPlotDialog)
        self.checkBox.setText(QtGui.QApplication.translate("AdcPlotDialog", "demodulate", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox.setObjectName(_fromUtf8("checkBox"))
        self.horizontalLayout_3.addWidget(self.checkBox)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.retranslateUi(AdcPlotDialog)
        QtCore.QMetaObject.connectSlotsByName(AdcPlotDialog)

    def retranslateUi(self, AdcPlotDialog):
        pass


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    AdcPlotDialog = QtGui.QDialog()
    ui = Ui_AdcPlotDialog()
    ui.setupUi(AdcPlotDialog)
    AdcPlotDialog.show()
    sys.exit(app.exec_())

