from __future__ import print_function
import numpy as np
import labrad
import twisted
import time
from   twisted.internet.defer import inlineCallbacks, returnValue
import labrad.units as U


def vec_phase(s):
    """
	:param s: vector
	:return:
	returns the phase in degrees of a two component vector
	"""
    phase = np.degrees(np.arctan2(s[1, 0], s[0, 0])) * (-1.0)
    # the -1.0 should be fixed in ac-box server/arduino code. It for some reason expects phase angle CW, not CCW
    if phase < 0: phase = 360 + phase
    return phase


class CapacitanceBridge:
    def __init__(self, cxn, ref2sample,  ref_ch="X2", f_slope=1, time_const=13, iterations=3,
                 tolerance=1 * U.uV, verbose=False, vsample=0.75):
        self.defaultUnit = U.mV
        self.scale_ratio = ref2sample
        self.channel = ref_ch
        self.tc = time_const
        self.slope = f_slope
        self.num_it = iterations
        self.client = cxn
        self.eps = tolerance[self.defaultUnit]
        self.connect = self.init_devices()
        self.responseMatrix = np.matrix(np.zeros((2, 2)))
        self.constantOffset = np.zeros((2, 1))
        self.vb = None
        self.vs = vsample
        self.verboseprint = print if verbose else lambda *a, **k: None

    def init_devices(self):
        b = False
        try:
            self.labrad_connect()
            self.lockin_connect()
            self.acbox_connect()
            b = True
        except Exception:
            print("Failed to initialize")
        return b

    def labrad_connect(self):
        try:
            if self.client is None:
                self.client = labrad.connect()
                print("connected")
        except twisted.internet.error.ConnectionRefusedError:
            print(".:CAP_BRIDGE-> Connection Failed: Check if LabRAD is running")

    def lockin_connect(self):
        try:
            self.lockin = self.client.amatek_7280_lock_in_amplifier
            self.lockin.select_device()
        except labrad.client.NotFoundError:
            print(
                    ".:CAP_BRIDGE-> Signal Recovery 7280 connection failed: Check if the device is connected and the server is running")

    def acbox_connect(self):
        print("connecting to acbox")
        try:
            self.acbox = self.client.ad5764_acbox
            self.acbox.select_device()
        except labrad.client.NotFoundError:
            print(
                    ".:CAP_BRIDGE-> AD5764 AC Box connection failed: Check if the device is connected and the server is running")

    def init_lockin(self):
        if self.connect:
            self.lockin.tc(self.tc)
            self.lockin.slope(self.slope)

    def balance(self, s_in1, s_in2, init_lockin=False):
        ''' Supplied s_in, s_in2 are vectors'''
        if init_lockin:
            self.init_lockin()

        res = self.get_linear_response_matrix(s_in1, s_in2)
        s_m1, self.responseMatrix = res[0], res[2]
        self.constantOffset = self.refine_balance(s_in1, s_m1)
        # print "First measured value and responseMatrix are\n{}\n {}".format(s_m1, self.responseMatrix)
        balanced = False
        i = 1
        while not balanced:
            v_b = self.find_balance()
            print("Iteration {}".format(i))
            if np.any(v_b > 1):
                print("Balance point out of range.")
                break
            r_b = np.linalg.norm(v_b)
            th_b = vec_phase(v_b)
            v_meas = self.measure(r_b, th_b)
            self.constantOffset = self.refine_balance(v_b, v_meas)
            if (i >= self.num_it):
                print("Hit max# of iterations")
                balanced = True
            if np.linalg.norm(v_meas) < self.eps:
                print("Hit Tolerance requirements: V_meas = {}, Tolerance set to: {}".format(np.linalg.norm(v_meas),
                                                                                             self.eps))
                balanced = True
            i += 1
        if balanced:
            self.vb = v_b
        return balanced

    def predict_measurement(self, s_in, offset=None):  # linear regime
        '''
		Using previously calculated response matrix M and constant vector offset(or offset provided) calculates the expected measurement values given an excitation s_in=(x,y).
		V_pred = M.S_in + V_off
		Returns column vector V_pred
		'''
        if offset is None:
            offset = self.constantOffset
        M = self.responseMatrix
        return M.dot(s_in) + offset

    def refine_balance(self, s_in, s_out):
        '''
		Using the previously calculated Response matrix M, determines offset by subtracting actually measured value and a predicted one.  
		Returns column vector V_off
		'''
        offset = np.zeros((2, 1))
        s_pred = self.predict_measurement(s_in, offset)
        c = s_out - s_pred
        return c  # new offset

    def find_balance(self, M=None, c=None):
        '''
		For response matrix M and offset V_off calculates balance point: v_out=0=M.V_b+c -> V_b = -M^(-1).c
		returns column vector
		'''
        if M is None:
            M = self.responseMatrix
        if c is None:
            c = self.constantOffset
        Balance = -1.0 * (M.I).dot(c)
        return Balance

    def calc_linear_response_matrix(self, s_out1, s_out2, s_in1, s_in2):
        '''
		For two given excitations s_in1, s_in2 and corresponding measurements s_out1, s_out2 calculates the response matrix. 
		M = ds_out/ds_in
		'''
        dS_in = s_in2 - s_in1
        dS_out = s_out2 - s_out1
        magnitude = np.linalg.norm(dS_out) / np.linalg.norm(dS_in)
        phase_shift = np.arctan2(dS_out[1, 0], dS_out[0, 0]) - np.arctan2(dS_in[1, 0], dS_in[0, 0])
        M = magnitude * np.matrix(
                ([np.cos(phase_shift), -1.0 * np.sin(phase_shift)], [np.sin(phase_shift), np.cos(phase_shift)]))
        return M

    def get_linear_response_matrix(self, s_in1, s_in2):
        '''
		Supply two excitations vectors and get a response matrix: the two excitations are applied to the chosen reference channel of the ac box.
		Measuring the lockin does auto sensitivity and auto gain adjustments.
		'''
        r1 = np.linalg.norm(s_in1)
        th1 = vec_phase(s_in1)

        r2 = np.linalg.norm(s_in2)
        th2 = vec_phase(s_in2)

        s_m1 = self.measure(r1, th1)
        s_m2 = self.measure(r2, th2)
        return s_m1, s_m2, self.calc_linear_response_matrix(s_m1, s_m2, s_in1, s_in2)

    def lockin_measure(self, auto=True):
        ''' Get measurement from lockin. if Auto = True start auto sensitivity. Currently waits 6 seconds as the worst case scenario/fail proof for the lockin to perform auto sensitivity
			otherwise the readout would be done before the sensitivity is set. 
			Returns vector column
		'''
        if auto:
            # print "auto sensitivity"
            self.lockin.set_auto_s()
            time.sleep(10)

        time.sleep(self.lockin.wait_time()['s'])  # wait settle time
        x = self.lockin.read_x()
        y = self.lockin.read_y()
        x = x[self.defaultUnit]
        y = y[self.defaultUnit]

        meas = np.array(([x], [y]))
        return meas

    def measure(self, voltage=None, phase=0):
        ''' measure systems response to reference excitation voltage.
			the excitation is provided as a voltage/phase pair
			if no voltage provided a measurement with current settings will be performed
			returns column vector of quadratures (x,y).Transpose
		'''
        if voltage is not None:
            if voltage <= 1:
                self.verboseprint("Setting (r,th) = {}".format(np.array((voltage, phase))))
                self.acbox.set_channel_voltage(self.channel, voltage)
                self.acbox.set_phase(phase)
            else:
                print("Couldn't set {} Overload.".format(voltage))

        meas = self.lockin_measure()
        self.verboseprint("Measured (x,y) = {}".format(meas.reshape(1, 2)[0]))
        return meas

    def get_physical_quantitites(self):
        """
		Maps the response matrix and constant offset to physical quantities -> Cs and Ds 
		Ds - dissipation of the sample 
		Cs - capacitance of the sample 
								  V_meas=(x,y)
					C_sample         ^            C_ref
					| |              |            | |
		  *------+--| |--+----+------*------------| |---------*
		V_sample |  | |  |    |      |            | |       V_ref
				 |       |    |      |
				 +--D_s--+    |    -----
							 D_p   ----- C_parasitic
							  |      |
							  +------+
									 *
									GND

		"""
        vb = self.vb
        vs = self.vs
        c_sample = -1.0 * vb[0, 0] / vs * self.scale_ratio
        d_sample = 1.0 * vb[1, 0] / vs * self.scale_ratio
        return np.array((c_sample, d_sample))

    def offbalance_scale(self):
        '''
		provided with the total capacitance and dissipation in the system, calculates their changes as a_function
		of the small deviations of measured vx, vy
		'''
        M = np.linalg.det(self.responseMatrix)
        vs = self.vs
        vbx = self.vb[0, 0]
        vby = self.vb[1, 0]
        vdx = self.constantOffset[0, 0]
        vdy = self.constantOffset[1, 0]

        Dtg = (vby * vdx - vbx * vdy) / (vbx ** 2 + vby ** 2) / M / vs * self.scale_ratio
        Ctg = (vbx * vdx + vby * vdy) / (vbx ** 2 + vby ** 2) / M * (-1.0) / vs * self.scale_ratio
        return Ctg, Dtg


if __name__ == '__main__':
    # cxn = labrad.connect(host='128.111.16.63', name="2D sweeper")
    cxn = labrad.connect()
    dv = cxn.data_vault
    dv.cd("Capacitance")

    vs_scale = 242.14 * U.mV
    refsc = 242.14 * U.mV
    ratio = refsc/vs_scale

    s1 = np.array((0.5, 0.5)).reshape(2, 1)
    s2 = np.array((-0.5, -0.5)).reshape(2, 1)

    cb = CapacitanceBridge(cxn, ratio, vsample=0.75,  ref_ch="Y2", time_const=16, iterations=14, tolerance=100 * U.uV, verbose=True)
    print(cb.balance(s1, s2))
    Cs, Ds = cb.get_physical_quantitites()
    print("Via balance: Cs = {}, Ds = {}".format(Cs, Ds))
    c_, d_ = cb.offbalance_scale()
    print("Scaling factors for offset: Ctg {} and Dtg {}".format(c_, d_))

    prompt = raw_input("Save Data? ([Y]/n)") or 'Y'
    if prompt.upper()=='Y':
        dv.new(raw_input("Enter File Name: "), ("C", "D", "c_", "d_"), ())
        dv.add_parameter("vs_scale", vs_scale)
        dv.add_parameter("ref scale", refsc)
        dv.add(Cs, Ds, c_, d_ )

