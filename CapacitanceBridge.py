# Version 0.1
import numpy as np
import time
'''
capacitance balancing bridge. supporting sr7280 and sr830
'''

class LinearBalancingBridge(object):
    """
    Find balance point given two input and two output values
    """
    def __init__(self, s_in1=None, s_in2=None, tolerance=None,
                iterations=None):
        """
        initialize balancing bridge object with two input and two output values
        -
        """
        self.s1 = s_in1
        self.s2 = s_in2
        self.tolerance = tolerance
        self.iterations = iterations

    def rebalance(self, vb, M, offset, itr):
        mb = self.measure(vb)
        self.balance(matrix=M, offset=offset, s1=vb, m1=mb, itr=itr)

    def predictMeasurement(self, s_in, offset=None):
        """
        Return measurement predictment for input vector
        """
        if offset is None:
            offset = self.constantOffset
        return self.M.dot(s_in) + offset

    def refineBalance(self, s1, m1):
        """
        Return the constant offset to bring the bridge to balance
        """
        offset = np.zeros((2, 1))
        s_pred = self.predictMeasurement(s1, offset)
        c = m1 - s_pred
        return c


    def findBalance(self, M=None, c=None):
        """
        Retun balance point vector v_b for a given Linear response matrix
        and constant offset.
        """
        if M is None:
            M = self.M
        if c is None:
            c = self.constantOffset
        return -1.0 * (M.I).dot(c)

    def responseMatrix(self, s1, s2, m1=None, m2=None):
        """
        Calculate the linear response matrix for the given input vectors.
        """
        if m1 is None:
            self.excite(s1)
            m1 = self.measure()
        if m2 is None:
            self.excite(s2)
            m2 = self.measure()

        dS_in = s2 - s1
        dS_out = m2 - m1
        magnitude = np.linalg.norm(dS_out) / np.linalg.norm(dS_in)
        phase_shift = np.arctan2(dS_out[1, 0] , dS_out[0, 0]) - np.arctan2(dS_in[1, 0], dS_in[0, 0])
        M = magnitude * np.matrix(([np.cos(phase_shift), -1.0 * np.sin(phase_shift)],
                                    [np.sin(phase_shift), np.cos(phase_shift)]))
        return M, m1, m2

    def balance(self, tol=None, itr=None, matrix=None, offset=None, s1=None, m1=None):
        """
        Balance until hit tolerance requirements or number of maximum number of
        iterations. If both aren't specified, find balance point.
        """
        if tol is None:
            tol = self.tolerance
        if itr is None:
            itr = self.iterations

        if matrix is None:
            self.excite(self.s1)
            m1 = self.measure()
            print "m1"
            print m1
            self.excite(self.s2)
            m2 = self.measure()
            self.M  = self.responseMatrix(self.s1, self.s2, m1, m2)[0]
        else:
            self.M = matrix

        print 'balance:'
        print 'matrix: '
        print self.M


        if offset is None:
            self.constantOffset = self.refineBalance(self.s1, m1)
        else:
            self.constantOffset = offset

        print 'offset'
        print self.constantOffset

        i = 1
        balanced = False
        while not balanced:
            print 'iteration'
            print i
            self.vb = self.findBalance()
            print 'vb'
            print self.vb
            if all(np.abs(self.vb) < 1):
                self.excite(self.vb)
                mb = self.measure()
                print 'mb'
                print mb
                self.constantOffset = self.refineBalance(self.vb, mb)
                print 'constantoffset'
                print self.constantOffset
                i+=1
                if np.linalg.norm(mb) < tol:
                    balanced = True
                    print("Balanced")
                elif  i > itr:
                    balanced = True
                    print("Hit maximum number of iterations {}".format(itr))
            else:
                print("Balanced point is out of range. Remove attenuators from ref.")
                balanced = True
                self.vb = np.array(([None], [None]))
        return self.vb


    def measure(self):
        pass

    def excite(self, s):
        pass


class CapacitanceBridge(LinearBalancingBridge):
    def __init__(self, acbox, excitation_chanel, lck, *args, **kwargs):
        super(CapacitanceBridge, self).__init__(*args, **kwargs)
        self.ac = acbox
        self.lck = lck
        self.chanel = excitation_chanel

    def capacitance(self, ac_scale):
        """
        Return sample capacitance and dissipation at balance point
        """
        vb = self.vb
        c_sample = -1.0 * vb[0, 0] * ac_scale
        d_sample = 1.0 * vb[1, 0] * ac_scale
        return np.array((c_sample, d_sample))

    def offBalance(self, ac_scale):
        """
        Return scaling factors for offbalance capacitance and dissipation
        """
        M = np.linalg.det(self.M)
        vbx, vby = np.array(self.vb).flatten()
        vdx, vdy = np.array(self.constantOffset).flatten()

        Dtg = ( vby * vdx - vbx * vdy) / (vbx ** 2 + vby ** 2) / M * ac_scale
        Ctg = -1.0*( vbx * vdx + vby * vdy) / (vbx ** 2 + vby **2 ) / M * ac_scale

        return Ctg, Dtg


class CapacitanceBridge7280Lockin(CapacitanceBridge):
    """
    Linear capacitance balancing bridge. Measurements are preformed with a
    Signal Recovery 7280 lockin amplifier. The AC excitation is provide by an
    AD**** AC box.
    """
    def __init__(self, time_const=None, *args, **kwargs):
        super(CapacitanceBridge7280Lockin, self).__init__(*args, **kwargs)
        if time_const is not None:
            self.lck.tc(time_const)
            time.sleep(lck.wait_time())

    def measure(self):
        lck = self.lockin
        lck.set_auto_s()
        time.sleep(10)

        time.sleep(lck.wait_time())
        x = lck.read_x()
        y = lck.read_y()
        return np.array(([x], [y]))

    def excite(self, s_in):
        ac = self.ac
        phase = self.vec_phase(s_in)
        ac.set_phase(phase)
        ac.set_voltage(self.chanel, np.linalg.norm(s_in))
        return True

    @staticmethod
    def vec_phase(s):
        phase = -1.0*np.degrees(np.arctan2(s[1, 0], s[0, 0]))
        if phase < 0:
            phase = 360 + phase
        return phase

class CapacitanceBridgeSR830Lockin(CapacitanceBridge):
    """
    Linear capacitance balancing bridge. Measurements are preformed with SRS
    SR830 lockin amplifier. The AC excitation is provide by an AD**** AC box.
    """
    def __init__(self, time_const=None, *args, **kwargs):
        super(CapacitanceBridgeSR830Lockin, self).__init__(*args, **kwargs)
        if time_const is not None:
            self.lck.time_constant(time_const)

    def measure(self):
        lck = self.lck
        wait_time = lck.wait_time()
        time.sleep(wait_time)
        lck.auto_sensitivity()
        x = lck.x()
        y = lck.y()
        meas = np.array(([x], [y]))
        return meas

    def excite(self, s_in):
        print 'Excite:'
        ac = self.ac
        phase = self.vec_phase(s_in)
        print 'phase: '
        print phase
        ac.set_phase(phase)
        ac.set_voltage(self.chanel, np.linalg.norm(s_in))
        return True

    def convertData(self, raw_meas, adc_offset=0, adc_scale=1, dac_offset=0, dac_expand=1)
        x, y = raw_meas
        fullscale = 10  
        lck = self.lck
        sen = lck.sensitivity()
        x = ((x/sen - dac_offset)*dac_expand*fullscale - adc_offset)*adc_scale
        y = ((y/sen - dac_offset)*dac_expand*fullscale - adc_offset)*adc_scale
        return x,y


    @staticmethod
    def vec_phase(s):
        phase = -1.0*np.degrees(np.arctan2(s[1, 0], s[0, 0]))
        if phase < 0:
            phase = 360 + phase
        return phase


if __name__ == '__main__':
    pass
