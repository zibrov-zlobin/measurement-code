'''
[info]
version = 2.0
'''

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import time
import math
import numpy as np
import labrad
import labrad.units as U
from include import CapacitanceBridge, NHMFLMagnetControl
import yaml

# TODO: make a gui.

RAMP_SPEED = 5.0  # volts per sec
RAMP_WAIT = 0.000  # seconds
X_MAX = 6.0
X_MIN = -6.0
Y_MAX = 4.0
Y_MIN = -5.0
ADC_CONVERSIONTIME = 250
ADC_AVGSIZE = 1

adc_offset = np.array([0.29391179, 0.32467712])
adc_slope = np.array([1.0, 1.0])
s1 = np.array((0.5, 0.5)).reshape(2, 1)
s2 = np.array((-0.5, -0.5)).reshape(2, 1)

magnet_query_time = 5.0


offbal = (-0.0, 0.0)
scale = 1.0

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

def lockin_select(cxn, s):
    if s == 'SR830':
        return cxn.sr830
    elif s == '7820':
        return cxn.amatek_7280_lock_in_amplifier

def init_acbox(acbox, stngs):
    vs_scale = 10**(-stngs['sample_atten']/20.0) * 250.0 * U.mV
    refsc = 10**(-stngs['ref_atten']/20.0) * 250.0 * U.mV
    ac_scale = (refsc / vs_scale)/float(stngs['chY1'])
    acbox.select_device()
    acbox.initialize(15)
    acbox.set_voltage("X1", stngs['chX1'])
    acbox.set_voltage("X2", stngs['chX1'])
    acbox.set_voltage("Y1", stngs['chY1'])
    acbox.set_voltage("Y2", stngs['chY2'])
    acbox.set_frequency(stngs['frequency'])
    time.sleep(1)
    return ac_scale

def init_bridge(lck, acbox, cfg):
    if cfg['lockin'] == 'SR830':
        bridge = CapacitanceBridge.CapacitanceBridgeSR830Lockin
    elif cfg['lockin'] == '7820':
        bridge = CapacitanceBridge.CapacitanceBridge7280Lockin

    stngs = cfg['balancing_settings']
    ref_ch = "Y"+str(stngs['ref_ch'])
    cb = bridge(lck=lck, acbox=acbox, time_const=stngs['balance_tc'],
                           iterations=stngs['iter'], tolerance=stngs['tolerance'],
                           s_in1=s1, s_in2=s2, excitation_channel=ref_ch)

    return cb

def rampfield(mag, field):
    print 'Ramping field to %1.5f ...' % field
    mag.setpoint(field)
    mag.ramp()
    status = mag.status()
    target_field = float(status['Setpoint'])
    actual_field = float(status['Field'])
    while abs(target_field - actual_field) > 1e-4:
        time.sleep(1)
        actual_field = float(status['Field'])
    print 'Target field reached.'

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

def create_file(dv, cfg, **kwargs): # try kwarging the vfixed
    try:
        dv.mkdir(cfg['file']['data_dir'])
        print "Folder {} was created".format(cfg['file']['data_dir'])
        dv.cd(cfg['file']['data_dir'])
    except Exception:
        dv.cd(cfg['file']['data_dir'])

    measurement = cfg['measurement']
    var_name1 = cfg[measurement]['v1']
    var_name2 = cfg[measurement]['v2']

    plot_parameters = {'extent': [cfg['meas_parameters']['n0_rng'][0],
                                  cfg['meas_parameters']['n0_rng'][1],
                                  cfg['meas_parameters']['b_rng'][0],
                                  cfg['meas_parameters']['b_rng'][1]],
                       'pxsize': [cfg['meas_parameters']['n0_pnts'],
                                  cfg['meas_parameters']['b_pnts']]
                      }

    if cfg['meas_parameters']['bscale']:
        if cfg['meas_parameters']['b_rng'][0] < cfg['meas_parameters']['b_rng'][1]:
            bf = cfg['meas_parameters']['b_rng']
            bscale = 1.0*bf[1]/bf[0] * cfg['meas_parameters']['scalefactor']
            plot_parameters['pxsize'][0] = int(plot_parameters['pxsize'][0] * bscale)
            plot_parameters['extent'][0] *= bscale
            plot_parameters['extent'][1] *= bscale
            print("The data will be scaled and will take more time; new plot extent and pxsize: {}".format(plot_parameters))

    if cfg['meas_parameters']['b_rng'][0] == cfg['meas_parameters']['b_rng'][1]:
        print("The start and final magnetic field match, Aligning?")
        ext_temp = max(abs(plot_parameters['extent'][0]), abs(plot_parameters['extent'][1]))
        plot_parameters['extent'][2] = ext_temp
        plot_parameters['extent'][3] = -1 * ext_temp
        print("Plot extent is:{}".format(plot_parameters['extent']))

    dv.new(cfg['file']['file_name']+"-plot", ("i", "j", var_name1, var_name2),
           ('Cs', 'Ds', 'B', 'N', 'X', 'Y', 't'))
    print("Created {}".format(dv.get_name()))
    dv.add_comment(cfg['file']['comment'])
    measurement_items = cfg[measurement].items()
    for parameter in range(len(measurement_items)):
        dv.add_parameter(measurement_items[parameter][0],measurement_items[parameter][1])
    dv.add_parameters(cfg['acbox_settings'].items())
    lockin_items = cfg['lockin_settings'].items()
    for parameter in range(len(lockin_items)):
        dv.add_parameter(lockin_items[parameter][0],lockin_items[parameter][1])
    balancing_items = cfg['balancing_settings'].items()
    for parameter in range(len(balancing_items)):
        dv.add_parameter(balancing_items[parameter][0],balancing_items[parameter][1])
    dv.add_parameter('n0_rng', cfg['meas_parameters']['n0_rng'])
    dv.add_parameter('b_pnts', cfg['meas_parameters']['b_pnts'])
    dv.add_parameter('n0_pnts', cfg['meas_parameters']['n0_pnts'])
    dv.add_parameter('b_rng', cfg['meas_parameters']['b_rng'])
    dv.add_parameter('extent', tuple(plot_parameters['extent']))
    dv.add_parameter('pxsize', tuple(plot_parameters['pxsize']))
    dv.add_parameter('plot', cfg['plot'])

    if kwargs is not None:
        for key, value in kwargs.items():
            dv.add_parameter(key, value)

def main():
    # Loads config
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    measurement = cfg['measurement']
    measurement_settings = cfg[measurement]
    balancing_settings = cfg['balancing_settings']
    lockin_settings = cfg['lockin_settings']
    acbox_settings = cfg['acbox_settings']
    dacadc_settings = cfg['dacadc_settings']
    meas_parameters = cfg['meas_parameters']
    delta_var = meas_parameters['delta_var']
    print delta_var

    # Connections and Instrument Configurations
    cxn = labrad.connect()
    reg = cxn.registry
    dv = cxn.data_vault
    dc = cxn.dac_adc
    mag = NHMFLMagnetControl.NHMFLMagnetControl()
    tc = cxn.lakeshore_372
    tc.select_device()
    lck = lockin_select(cxn, cfg['lockin'])
    lck.select_device()
    acbox = cxn.acbox
    ac_scale = init_acbox(acbox, acbox_settings)

    quad_dc = cxn.dcbox_quad_ad5780
    quad_dc.select_device()

    v_fixed = float(quad_dc.get_voltage(1)) - meas_parameters['zOffset']  # this might need to have a case structure based on whats fixed?
    print("Fixed TM gating voltage is {}".format(v_fixed))

    dc.select_device()
    dc.set_conversiontime(measurement_settings['read1'], ADC_CONVERSIONTIME)
    dc.set_conversiontime(measurement_settings['read2'], ADC_CONVERSIONTIME)

    reg.cd(['Measurements', 'Capacitance'])
    rebalance = balancing_settings['rebalance']
    if rebalance:
        ch_x = measurement_settings['ch1']
        ch_y = measurement_settings['ch2']

        cb = init_bridge(lck, acbox, cfg)
        v1_balance, v2_balance = function_select(measurement_settings['fixed'])(balancing_settings['p0'], balancing_settings['n0'], meas_parameters['delta_var'], v_fixed)

        lck.time_constant(balancing_settings['balance_tc'])
        dc.set_voltage(ch_x, v1_balance)
        dc.set_voltage(ch_y, v2_balance)

        print cb.balance()
        cs, ds = cb.capacitance(ac_scale)
        print("Via balance: Cs = {}, Ds = {}".format(cs, ds))
        c_, d_ = cb.offBalance(ac_scale)
        print("Scaling factors for offset: Ctg {} and Dtg {}".format(c_, d_))
        reg.set('capacitance_params', [('cs', cs), ('ds', ds), ('c_', c_), ('d_', d_)])

        vb = cb.vb # this is the balance point. this is a vector with (x,y) -> amplitude sqrt(x**2+y**2), phase (atan y/x)
        magnitude = np.sqrt(vb[0]**2 + vb[1]**2)
        phase = np.degrees(np.arctan2(vb[1], vb[0])) * (-1.0)
        if phase < 0: phase = 360 + phase
        reg.set('acbox_params', (magnitude, phase))
    else:
        acbox_params = reg.get('acbox_params')
        ref_ch = "Y"+str(balancing_settings['ref_ch'])
        acbox.set_voltage(ref_ch, acbox_params[0][0][0])
        acbox.set_phase(acbox_params[1][0][0])

        cap_params = reg.get('capacitance_params')
        cs = cap_params[0][1]
        ds = cap_params[1][1]
        c_ = cap_params[2][1]
        d_ = cap_params[3][1]
        print("Cs = {}, Ds = {}".format(cs, ds))
        print("Scaling factors for offset: Ctg {} and Dtg {}".format(c_, d_))

    capacitance_params = {'Capacitance': cs, 'Dissipation': ds,
                          'offbalance c_': c_, 'offbalance d_': d_}

    probe = tc.probe()
    mc = tc.mc()
    p0fixed = meas_parameters['p0fixed']

    create_file(dv, cfg, **dict({'p0fixed': p0fixed, 'vfixed': v_fixed, 'temperature_probe': probe, 'temperature_magnet_chamber': mc}, **capacitance_params))

    #ac_gain_var = int(round(lockin_settings['acgain'] / 6.666))
    tc_var = lockin_settings['tc']
    sens_var = lockin_settings['sensitiviy']
    lck.time_constant(tc_var)
    #lck.set_ac_gain(ac_gain_var)
    lck.sensitivity(sens_var)

    s = lck.sensitivity()
    time.sleep(.25)
    t0 = time.time()

    pxsize = (meas_parameters['n0_pnts'], meas_parameters['b_pnts'])
    extent = (meas_parameters['n0_rng'][0], meas_parameters['n0_rng'][1], meas_parameters['b_rng'][0], meas_parameters['b_rng'][1])
    num_x0 = pxsize[0]
    num_y = pxsize[1]
    print extent, pxsize, p0fixed

    DELAY_MEAS = 3 * lockin_settings['tc'] * 1e6
    est_time = (pxsize[0] * pxsize[1] + pxsize[1]) * DELAY_MEAS * 1e-6 / 60.0
    dt = pxsize[0]*DELAY_MEAS*1e-6/60.0
    print("Will take a total of {} mins. With each line trace taking {} ".format(est_time, dt))


    field = np.linspace(extent[2], extent[3], num_y)  # generate the array of fields

    dac_ch1 = measurement_settings['ch1']
    dac_ch2 = measurement_settings['ch2']
    adc_ch1 = measurement_settings['read1']
    adc_ch2 = measurement_settings['read2']


    for i in range(num_y):
        #rampfield(mag, field[i])

        if meas_parameters['bscale']:
            bscale = field[i]*1.0/field[0]*meas_parameters['scalefactor']
        else:
            bscale = 1

        num_x = int(num_x0*bscale)
        ext_x = (extent[0]*bscale, extent[1]*bscale)

        m, mdn = mesh(vfixed=v_fixed, offset=(0, -0.0), drange=(p0fixed, p0fixed),
                      nrange=(ext_x[0], ext_x[1]), fixed=measurement_settings['fixed'],
                      pxsize=(num_x, num_y), delta=delta_var)

        data_x = np.zeros(num_x)
        data_y = np.zeros(num_x)

        vec_x = m[i, :][:, 0]
        vec_y = m[i, :][:, 1]

        # vec_x = (m[i, :][:, 0] - dacadc_settings['ch1_offset'])
        # vec_y = (m[i, :][:, 1] - dacadc_settings['ch2_offset'])

        md = mdn[i, :][:, 0]
        mn = mdn[i, :][:, 1]

        mask = np.logical_and(np.logical_and(vec_x <= X_MAX, vec_x >= X_MIN),
                              np.logical_and(vec_y <= Y_MAX, vec_y >= Y_MIN))
        if np.any(mask == True):
            start, stop = np.where(mask == True)[0][0], np.where(mask == True)[0][-1]

            num_points = stop - start + 1
            # print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print("{} of {}  --> Ramping. Points: {}".format(i + 1, num_y, num_points))
            dc.buffer_ramp([dac_ch1, dac_ch2],
                           [adc_ch1, adc_ch2],
                           [vec_x[start], vec_y[start]],
                           [vec_x[stop], vec_y[stop]],
                           num_points, DELAY_MEAS, ADC_AVGSIZE)

            d_read = dc.serial_poll.future(2, num_points)
            d_tmp = d_read.result()

            data_x[start:stop + 1], data_y[start:stop + 1] = d_tmp
            data_x[start:stop + 1] = (data_x[start:stop + 1] - adc_offset[0]) / adc_slope[0] / 2.5 * s
            data_y[start:stop + 1] = (data_y[start:stop + 1] - adc_offset[1]) / adc_slope[1] / 2.5 * s

        d_cap = (c_ * data_x + d_ * data_y) + cs
        d_dis = (d_ * data_x - c_ * data_y) + ds

        j = np.linspace(0, num_x - 1, num_x)
        ii = np.ones(num_x) * i
        t1 = np.ones(num_x) * time.time() - t0
        mB = np.ones(num_x) * field[i]
        totdata = np.array([j, ii, vec_x, vec_y, d_cap, d_dis, mB, mn, data_x, data_y, t1])
        dv.add(totdata.T)
    print("it took {} s. to write data".format(time.time() - t0))


if __name__ == '__main__':
    main()
