import numpy as np
import labrad
import labrad.units as U
import time
import math
from CapacitanceBridge import CapacitanceBridge

# TODO: make a gui.

data_dir = "SZ11"
file_name = "Csym_10T"
comment = "Csym, 100 mV bias"

Cpen = {'v1': 'V_tg [V]', 'ch1': '1',
        'v2': 'V_sample [V]', 'ch2': '0',
        'read1': '0',
        'read2': '1',
        'fixed': 'vb'

        }

Csym = {'v1': 'V_tg [V]', 'ch1': '1',
        'v2': 'V_bg [V]', 'ch2': '0',
        'read1': '0',
        'read2': '1',
        'fixed': 'vs'
        }

measurement = Csym

lockin_settings = {'tc': 0.02,
                   'acgain': 6,
                   'sensitivity': 1e-3
                   }

balancing_settings = {'balance_tc': 0.05,
                      'n0': 0.5,
                      'p0': 0,
                      'tolerance': 500.0,
                      'ref_ch': 2  # Y2
                      }

acbox_settings = {'chX1': 1.0,
                  'chX2': 1.0,
                  'chY1': 0.75,
                  'chY2': 0.75,
                  'frequency': 67732.77,
                  'ref_atten': 23,
                  'sample_atten': 30
                  }

nhmfl_env = {'bfield': 0,
             'brate': 0,
             't_cernox': 0.340,
             't_ruox': 0.00
             }

x2amp = {'ch1 offset': -0.0024784313,
         'ch1 scale': 0.9999789608,
         'ch2 offset': -0.0043156862,
         'ch2 scale': 0.99986735604}

# x2amp = {'ch1 offset': -0.0008,
#          'ch1 scale': 2.0008,
#          'ch2 offset': -0.0065,
#          'ch2 scale': 2.0004}

var_pxsize = np.array([(200, 100)])
var_extent = np.array([(-2-0.6, 2-0.6, -2-1, 2-1)])  # -5.3, -4.55)


delta_var = 0.0575
offbal = (-0.0, 0.0)
scale = 1.0
brate = nhmfl_env['brate']  # T/min
b0 = nhmfl_env['bfield']

ch1_scale = x2amp['ch1 scale']
ch1_offset = x2amp['ch1 offset']
ch2_scale = x2amp['ch2 scale']
ch2_offset = x2amp['ch2 offset']


tau = lockin_settings['tc']
gaindB = lockin_settings['acgain']
sens = lockin_settings['sensitivity']
DELAY_MEAS = 3 * tau * 1e6  # delay between changing dc voltages

balance_tc = balancing_settings['balance_tc']
REF_CHN = "Y" + str(balancing_settings['ref_ch'])
tol = balancing_settings['tolerance'] * U.uV

nbal = balancing_settings['n0']
dbal = balancing_settings['p0']

sample_attendB = acbox_settings['sample_atten']
ref_attendB = acbox_settings['ref_atten']
ac_freq = acbox_settings['frequency']

vs_scale = 10 ** (-sample_attendB / 20.0) * 250.0 * U.mV  #
refsc = 10 ** (-ref_attendB / 20.0) * 250.0 * U.mV
ratio = refsc / vs_scale

var_name1 = measurement['v1']
var_name2 = measurement['v2']
var_ch1 = measurement['ch1']
var_ch2 = measurement['ch2']
read_ch1 = measurement['read1']  # x ch
read_ch2 = measurement['read2']  # y ch

RAMP_SPEED = 5.0  # volts per sec
RAMP_WAIT = 0.000  # seconds
X_MAX = 16.0
X_MIN = -16.0
Y_MAX = 16.0
Y_MIN = -16.0
ADC_CONVERSIONTIME = 250
ADC_AVGSIZE = 1

adc_offset = np.array([0.29391179, 0.32467712])
adc_slope = np.array([1.0, 1.0])
s1 = np.array((0.5, 0.5)).reshape(2, 1)
s2 = np.array((-0.5, -0.5)).reshape(2, 1)

magnet_query_time = 5.0




def vb_fixed(p0, n0, delta, vb):
    """
    :param p0: polarizing field
    :param n0: charge carrier density
    :param delta: capacitor asymmetry
    :param vb: fixed voltage set on the bottom gate
    :return: (v_top, v_sample)
    """
    return vb - (n0 * delta - p0) / (1.0 - delta ** 2), vb - 0.5 * (n0 - p0) / (1.0 - delta)


def vt_fixed(p0, n0, delta, vt):
    """
    :param p0: polarizing field
    :param n0: charge carrier density
    :param delta: capacitor asymmetry
    :param vt: fixed voltage set on the top gate
    :return: (v_bot, v_sample)
    """
    return (n0 * delta - p0) / (1.0 - delta ** 2) + vt, vt - 0.5 * (n0 + p0) / (1.0 + delta)


def vs_fixed(p0, n0, delta, vs):
    """
    :param p0: polarizing field
    :param n0: charge carrier density
    :param delta: capacitor asymmetry
    :param vs: fixed voltage set on graphene sample
    :return: (v_top, v_bottom)
    """
    return vs + 0.5 * (n0 + p0) / (1.0 + delta), vs + 0.5 * (n0 - p0) / (1.0 - delta)



def function_select(s):
    """
    :param s: ('vb', 'vt', 'vs') selection based on which parameter is fixed
    :return: function f
    """
    if s == 'vb':
        f = vb_fixed

    elif s == 'vt':
        f = vt_fixed
    elif s == 'vs':
        f = vs_fixed
    return f


def dac_adc_measure(dacadc, scale, chx, chy):
    return np.array([dacadc.read_voltage(chx), dacadc.read_voltage(chy)]) / 2.5 * scale


def mesh(vfixed, offset, drange, nrange, fixed="vb", pxsize=(100, 100), delta=0.0):
    """
    drange and nrange are tuples (dmin, dmax) and (nmin, nmax)
    offset  is a tuple of offsets:  (N0, D0)
    pxsize  is a tuple of # of steps:  (N steps, D steps)
    fixed sets the fixed channel: "vb", "vt", "vs"
    fast  - fast axis "D" or "N"
    """
    f = function_select(fixed)
    p0 = np.linspace(drange[0], drange[1], pxsize[1]) - offset[1]
    n0 = np.linspace(nrange[0], nrange[1], pxsize[0]) - offset[0]
    n0, p0 = np.meshgrid(n0, p0)  # p0 - slow n0 - fast
    # p0, n0 = np.meshgrid(p0, n0)  # p0 - slow n0 - fast
    v_fast, v_slow = f(p0, n0, delta, vfixed)
    return np.dstack((v_fast, v_slow)), np.dstack((p0, n0))


def tm_tuneup(dc, read_voltage, gate_voltages):
    print("started tune up")
    drain = float(dc.get_voltage(0))
    target = drain /2.0/100
    drain = np.zeros(gate_voltages.size)
    for i in range(gate_voltages.size):
        dc.set_voltage(1, gate_voltages[i])
        tmp = read_voltage()
        if type(tmp) is labrad.units.Value:
            tmp = tmp['V']
        drain[i] = tmp

    idx = np.argmin(np.absolute(drain-target))
    gate = gate_voltages[idx]
    dc.set_voltage(1, gate)
    read_voltage()
    return gate

def main():
    cxn = labrad.connect()
    dv = cxn.data_vault
    dc = cxn.dac_adc
    # mag = cxn.ami_430
    # tc = cxn.lakeshore_372
    lck = cxn.amatek_7280_lock_in_amplifier
    acbox = cxn.acbox

    acbox.select_device()
    acbox.initialize(15)
    acbox.set_voltage("X1", acbox_settings['chX1'])
    acbox.set_voltage("X2", acbox_settings['chX1'])
    acbox.set_voltage("Y1", acbox_settings['chY1'])
    acbox.set_voltage("Y2", acbox_settings['chY2'])
    acbox.set_frequency(ac_freq)
    time.sleep(1)

    quad_dc = cxn.dcbox_quad_ad5780;
    quad_dc.select_device()


    v_fixed = float(quad_dc.get_voltage(1))
    print("Fixed TM gating voltage is {}".format(v_fixed))




    vs_scale = 10**(-acbox_settings['sample_atten']/20.0) * 250.0 * U.mV
    refsc = 10**(-acbox_settings['ref_atten']/20.0) * 250.0 * U.mV
    ratio = refsc / vs_scale

    dc.select_device()
    dc.set_conversiontime(int(read_ch1), ADC_CONVERSIONTIME)
    dc.set_conversiontime(int(read_ch2), ADC_CONVERSIONTIME)


    lck.select_device()



    ch_x = int(var_ch1)
    ch_y = int(var_ch2)

    try:
        dv.mkdir(data_dir)
        print "Folder {} was created".format(data_dir)
        dv.cd(data_dir)
    except Exception:
        dv.cd(data_dir)

    iter = 4
    cb = CapacitanceBridge(cxn, ratio, ref_ch=REF_CHN, time_const=18,
                           iterations=iter, tolerance=tol, verbose=True, vsample=acbox_settings['chY1'])

    v1_balance, v2_balance = function_select(measurement['fixed'])(dbal, nbal, delta_var, v_fixed)

    lck.tc(int(round(3 * math.log(balance_tc, 10) + 18)))
    dc.set_voltage(ch_x, v1_balance)
    dc.set_voltage(ch_y, v2_balance)

    ac_gain_var = int(round(gaindB / 6.666))
    tc_var = int(round(3 * math.log(tau, 10) + 18))
    sens_var = int(round(3 * math.log(sens, 10) + 27))
    lck.tc(tc_var)
    lck.set_ac_gain(ac_gain_var)
    lck.set_sensitivity(sens_var)

    s = lck.get_sensitivity()['mV']
    time.sleep(.25)
    t0 = time.time()

    print cb.balance(s1, s2)
    cs, ds = cb.get_physical_quantitites()
    print("Via balance: Cs = {}, Ds = {}".format(cs, ds))
    c_, d_ = cb.offbalance_scale()
    print("Scaling factors for offset: Ctg {} and Dtg {}".format(c_, d_))

    capacitance_params = {'Capacitance': cs, 'Dissipation': ds, 'offbalance c_': c_, 'offbalance d_': d_}

    ac_gain_var = int(round(gaindB / 6.666))
    tc_var = int(round(3 * math.log(tau, 10) + 18))
    sens_var = int(round(3 * math.log(sens, 10) + 27))
    lck.tc(tc_var)
    lck.set_ac_gain(ac_gain_var)
    lck.set_sensitivity(sens_var)

    s = lck.get_sensitivity()['mV']
    time.sleep(.25)
    t0 = time.time()



    for k in range(var_extent.shape[0]):
        extent = var_extent[k]
        pxsize = var_pxsize[k]
        num_x = pxsize[0]
        num_y = pxsize[1]


        print extent, pxsize

        dv.new(file_name+"-plot", ("i", "j", var_name1, var_name2), ('Cs', 'Ds', 'D', 'N', 'X', 'Y', 'B', 't'))
        est_time = (pxsize[0] * pxsize[1] + pxsize[1]) * DELAY_MEAS * 1e-6 / 60.0
        dt = pxsize[0]*DELAY_MEAS*1e-6/60.0
        print("Created {}. Will take a total of {} mins. With each line trace taking {} ".format(dv.get_name(), est_time, dt))
        dv.add_comment(comment)
        dv.add_parameters(measurement.items())
        dv.add_parameters(acbox_settings.items())
        dv.add_parameters(lockin_settings.items())
        dv.add_parameters(balancing_settings.items())
        dv.add_parameters(capacitance_params.items())
        # dv.add_parameter('Mixing Chamber Temperature', t_mc)
        # dv.add_parameter('Probe Temperature', t_probe)
        dv.add_parameter('Magnetic Field', nhmfl_env['bfield'])
        dv.add_parameter('Probe Temperature RuOx', nhmfl_env['t_ruox'])
        dv.add_parameter('Probe Temperature Cernox', nhmfl_env['t_cernox'])
        dv.add_parameter('extent', extent)
        dv.add_parameter('pxsize', pxsize)
        dv.add_parameters(x2amp.items())
        dv.add_parameter('vfixed', v_fixed)




        m, mdn = mesh(vfixed=v_fixed, offset=(0, -0.0), drange=(extent[2], extent[3]), nrange=(extent[0], extent[1]),
                          fixed=measurement['fixed'], pxsize=pxsize, delta=delta_var)


        for i in range(num_y):



            data_x = np.zeros(num_x)
            data_y = np.zeros(num_x)
            bfields = np.zeros(num_x)


            # vec_x = m[i, :][:, 0]*scale
            # vec_y = m[i, :][:, 1]*scale

            vec_x = (m[i, :][:, 0] - ch1_offset)/ ch1_scale
            vec_y = (m[i, :][:, 1] - ch2_offset) / ch2_scale

            md = mdn[i, :][:, 0]
            mn = mdn[i, :][:, 1]

            mask = np.logical_and(np.logical_and(vec_x <= X_MAX, vec_x >= X_MIN),
                                    np.logical_and(vec_y <= Y_MAX, vec_y >= Y_MIN))
            if np.any(mask == True):
                start, stop = np.where(mask == True)[0][0], np.where(mask == True)[0][-1]

                num_points = stop - start + 1
                print(time.strftime("%Y-%m-%d %H:%M:%S"))
                print("{} of {}  --> Ramping. Points: {}".format(i + 1, num_y, num_points))

                dc.buffer_ramp([int(var_ch1), int(var_ch2)], [int(read_ch1), int(read_ch2), 2],
                                [vec_x[start], vec_y[start]],
                                [vec_x[stop], vec_y[stop]],
                                num_points, DELAY_MEAS, ADC_AVGSIZE)

                d_read = dc.serial_poll.future(3, num_points)
                d_tmp = d_read.result()

                data_x[start:stop + 1], data_y[start:stop + 1], bfields[start:stop + 1] = d_tmp
                data_x[start:stop + 1] = (data_x[start:stop + 1] / 2.5 * s - adc_offset[0]) / adc_slope[0]
                data_y[start:stop + 1] = (data_y[start:stop + 1] / 2.5 * s - adc_offset[1]) / adc_slope[1]

            d_cap = (c_ * data_x + d_ * data_y) + cs
            d_dis = (d_ * data_x - c_ * data_y) + ds

            j = np.linspace(0, num_x - 1, num_x)
            ii = np.ones(num_x) * i
            t1 = np.ones(num_x) * time.time() - t0
            totdata = np.array([j, ii, vec_x, vec_y, d_cap, d_dis, md, mn, data_x, data_y, bfields, t1])
            dv.add(totdata.T)
        print("it took {} s. to write data".format(time.time() - t0))


if __name__ == '__main__':
    main()
