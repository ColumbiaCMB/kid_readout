__author__ = 'gjones'

from kid_readout.roach.heterodyne import RoachHeterodyne

class TestHeterodyne():
    @classmethod
    def setup_class(cls):
        print "********** making roach"
        cls.ri = RoachHeterodyne()
        cls.ri.initialize(use_config=False)
    def setup(self):
        print "blanking roach"
        self.ri.r.write_int('sync',0)
    def test_1(self):
        print "reading test_1",self.ri.r.read_int('sync')
        self.ri.r.write_int('sync',2)
    def test_2(self):
        print "reading test_2",self.ri.r.read_int('sync')
        self.ri.r.write_int('sync',2)