import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from twisted.internet.defer import inlineCallbacks
import numpy as np

DIR_PLOT = "HZS63"

class MainWindow(QtGui.QMainWindow):
    """
    Plots data obtained from a 2d Sweeper
    Make more generic - > parse tags, parse attribute for live plotting,
    specify the columns, labels and titles.
    """
    ID_NEWSET = 00001
    ID_NEWDATA = 00002
    ID_NEWPARAM = 00003

    def __init__(self, reactor, parent=None):
        """
        2D dc sweeper grpaher
        :param reactor:
        :param parent:
        :return:
        """
        super(MainWindow, self).__init__(parent)
        self.reactor = reactor
        self.setGeometry(200, 200, 1600, 880)
        self.setWindowTitle("Plotter 2D/1D")
        self.sweeper_widget = Sweep2DWidget()
        self.setCentralWidget(self.sweeper_widget)
        self.win1 = self.sweeper_widget.win1
        self.win2 = self.sweeper_widget.win2
        self.Window_1Dplot = LineTraces()
        self.Window_1Dplot.show()
        self.cap1d = self.Window_1Dplot.cap
        self.dis1d = self.Window_1Dplot.dis
        self.dvconnect()


    @inlineCallbacks
    def dvconnect(self):
        """

        :return:
        """
        from labrad.wrappers import connectAsync
        cxn = yield connectAsync(name='plotter2d1d')
        self.dv = cxn.data_vault
        yield self.dv.signal__new_dataset(self.ID_NEWSET)
        yield self.dv.addListener(listener=self.open_dataset, source=None, ID=self.ID_NEWSET)
        yield self.dv.signal__data_available(self.ID_NEWDATA)
        yield self.dv.addListener(listener=self.update, source=None, ID=self.ID_NEWDATA)
        yield self.dv.signal__new_parameter(self.ID_NEWPARAM)
        yield self.dv.addListener(listener=self.update_params, source=None, ID=self.ID_NEWPARAM)
        yield self.dv.cd(DIR_PLOT)

    def open_dataset(self, cntx, signal):
        print "Data created"
        if signal[-4:]=='plot':
            self.dv.open(signal)
            self.dv.parameters()
            self.dv.get_parameters()
            print "set opened"

    def create_graph(self):
        self.dx = np.zeros(self.pxsize)
        self.dy = np.zeros(self.pxsize)
        x0, x1 = (self.extent[0], self.extent[1])
        y0, y1 = (self.extent[2], self.extent[3])
        xscale, yscale = 1.0*(x1-x0) / self.dx.shape[0], 1.0 * (y1-y0) / self.dx.shape[1]
        self.win1.setImage(self.dx, pos=[x0, y0], scale=[xscale, yscale])
        self.win2.setImage(self.dy, pos=[x0, y0], scale=[xscale, yscale])
        self.plot1 = pg.PlotCurveItem(x=[], y=[])
        self.plot2 = pg.PlotCurveItem(x=[], y=[])
        pen1 = pg.mkPen('y', width=1.5)
        pen2 = pg.mkPen('c', width=1.5)
        self.plot1.setPen(pen1)
        self.plot2.setPen(pen2)
        self.cap1d.clear()
        self.dis1d.clear()
        self.cap1d.addItem(self.plot1)
        self.dis1d.addItem(self.plot2)
        self.cap = []
        self.dissip = []
        self.volt = np.zeros(self.pxsize)


    @inlineCallbacks
    def update(self, cntx, signal):
        data = yield self.dv.get()
        d = np.array(data)
        for i in range(d.shape[0]):
            self.dx[d[i, 0], d[i, 1]] = d[i, 4]
            self.dy[d[i, 0], d[i, 1]] = d[i, 5]
            self.volt[d[i, 0], d[i, 1]] = d[i, 7]
            if (d[i, 0] == 0)and(d[i, 1] == 0):
                self.dx.fill(d[i, 4])
                self.dy.fill(d[i, 5])

        ii = d[i, 0] # this is the last element in the horizontal trace.
        jj = d[i, 1] # this is the last horizontal trace

        x0, x1 = (self.extent[0], self.extent[1])
        y0, y1 = (self.extent[2], self.extent[3])
        xscale, yscale = 1.0*(x1-x0) / self.dx.shape[0], 1.0 * (y1-y0) / self.dx.shape[1]
        self.win1.setImage(self.dx, pos=[x0, y0], scale=[xscale, yscale])
        self.win2.setImage(self.dy, pos=[x0, y0], scale=[xscale, yscale])
        self.plot1.clear()
        self.plot2.clear()
        self.plot1.setData(x=self.volt[:ii, jj], y=self.dx[:ii, jj])
        self.plot2.setData(x=self.volt[:ii, jj], y=self.dy[:ii, jj])
        # self.plot2 = self.dis1d.addItem(pg.PlotCurveItem(x=volt, y=dis))
        # self.plot1 = self.cap1d.getPlotItem()

    def closeEvent(self, e):
        self.reactor.stop()
        print "stop"

    @inlineCallbacks
    def update_params(self, cntx, signal):
        print "params updated"
        params = yield self.dv.get_parameters()
        print params
        for p in params:
            if 'pxsize' in p:
                self.pxsize = dict(params)['pxsize']
            if 'extent' in p:
                self.extent = dict(params)['extent']
        if (hasattr(self, 'extent'))and(hasattr(self, 'pxsize')):
            print "creating graph"
            self.create_graph()

class Sweep2DWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(Sweep2DWidget, self).__init__(parent)
        # self.layout = QtGui.QHBoxLayout(self)
        self.layout = QtGui.QGridLayout(self)
        # self.caplayout = QtGui.QVBoxLayout(self)
        self.view1 = pg.PlotItem(title="Capacitance", labels={'right': "V1", 'bottom': "V2",
                                                                                'top': "V2", 'left': "V1"})
        self.w1autoLevels = True
        self.w2autoLevels = True

        self.chBx_autolevels1 = QtGui.QCheckBox('Auto Levels')
        self.chBx_autolevels1.setChecked(True)
        # self.chBx_autolevels1.stateChanged.connect()
        self.view1.setAspectLocked(False)
        self.win1 = pg.ImageView(view=self.view1)
        self.view1.setAspectLocked(False)


        self.layout.addWidget(self.win1, *(0,0))
        self.layout.addWidget(self.chBx_autolevels1, *(1,0))

        # self.dislayout = QtGui.QVBoxLayout(self)
        self.view2 = pg.PlotItem(title="Dissipation", labels={'right': "V1", 'bottom': "V2",
                                                                                'top': "V2", 'left': "V1"})

        self.view2.setAspectLocked(False)
        self.chBx_autolevels2 = QtGui.QCheckBox('Auto Levels')
        self.win2 = pg.ImageView(view=self.view2)
        self.view2.setAspectLocked(False)
        self.layout.addWidget(self.win2, *(0,1))
        self.layout.addWidget(self.chBx_autolevels2, *(1,1))


        # self.layout.addWidget(self.win1)

        # self.win2 = pg.ImageView(view=self.view2)
        # self.layout.addWidget(self.win2)

        self.win1.ui.roiBtn.hide()
        self.win1.ui.menuBtn.hide()
        self.win2.ui.roiBtn.hide()
        self.win2.ui.menuBtn.hide()
        self.win1.ui.histogram.item.gradient.loadPreset('bipolar')
        self.win2.ui.histogram.item.gradient.loadPreset('bipolar')
        # self.win1.ui.histogram.item.region.hide()
        self.setLayout(self.layout)

    def enable_autolevels(self, checkbox):
        if checkbox == 1:
            pass


class LineTraces(QtGui.QWidget):
    def __init__(self, parent=None):
        super(LineTraces, self).__init__(parent)
        self.layout = QtGui.QVBoxLayout(self)
        self.cap = pg.PlotWidget(title="Capacitance", labels={'left': "V1", 'bottom': "V2"})
        self.dis = pg.PlotWidget(title="Dissipation", labels={'left': "V1", 'bottom': "V2"})
        self.layout.addWidget(self.cap)
        self.layout.addWidget(self.dis)
        self.setLayout(self.layout)
        self.setWindowTitle("Plotter 1D")



if __name__=="__main__":
    a = QtGui.QApplication( [] )
    import qt4reactor
    qt4reactor.install()
    from twisted.internet import reactor

    window = MainWindow(reactor)
    window.show()

    # win1d =  SecondWindow(reactor)
    # win1d.show()
    reactor.run()
