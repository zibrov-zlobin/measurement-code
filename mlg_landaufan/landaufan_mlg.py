# Script to measure a stripe Landau Fan of n0 in MLG.

import numpy as np
import labrad, time, math
import labrad.units as U
from CapacitanceBridge import CapacitanceBridge
import yaml


DELAY_MEAS = 3 * tau * 1e6  # delay between changing dc voltages

RAMP_SPEED = 5.0  # volts per sec
RAMP_WAIT = 0.000  # seconds
X_MAX = 5.0
X_MIN = -5.0
ADC_CONVERSIONTIME = 250
ADC_AVGSIZE = 1

adc_offset = np.array([0.29391179, 0.32467712])
adc_slope = np.array([1.0, 1.0])
s1 = np.array((0.5, 0.5)).reshape(2, 1)
s2 = np.array((-0.5, -0.5)).reshape(2, 1)

def vb_fixed(n0, vb):
    return n0 - vb

def vt_fixed(n0, vt):
    return -n0 + vt

def function_select(s):
    if s == 'vb':
        f = vb_fixed
    elif s == 'vt':
        f = vt_fixed
    return f

def init_acbox(acbox, stngs):
    vs_scale = 10**(-stngs['sample_atten']/20.0) * 250.0 * U.mV
    refsc = 10**(-stngs['ref_atten']/20.0) * 250.0 * U.mV
    ratio = refsc / vs_scale
    acbox.select_device()
    acbox.initialize(15)
    acbox.set_voltage("X1", stngs['chX1'])
    acbox.set_voltage("X2", stngs['chX1'])
    acbox.set_voltage("Y1", stngs['chY1'])
    acbox.set_voltage("Y2", stngs['chY2'])
    acbox.set_frequency(ac_freq)
    time.sleep(1)
    return ratio

def init_bridge(cxn, ratio, cfg):
    stngs = cfg['balancing_settings']
    ref_ch = "Y"+str(stngs['ref_ch'])
    cb = CapacitanceBridge(cxn, ratio, ref_ch=ref_ch, time_const=stngs['tc'],
                           iterations=stngs['iter'], tolerance=stngs['tolerance'],
                            verbose=True, vsample=cfg['acbox_settings']['chY1'])

    return cb

def create_file(dv, cfg, **kwargs): # try kwarging the vfixed
    try:
        dv.mkdir(cfg['file']['data_dir'])
        print "Folder {} was created".format(cfg['file']['data_dir'])
        dv.cd(cfg['file']['data_dir'])
    except Exception:
        dv.cd(cfg['file']['data_dir'])

    var_name1 = cfg['measurement']['v1']
    var_name2 = cfg['measurement']['v2']

    plot_parameters = {'extent': [cfg['meas_parameters']['n0_rng'][0],
                                  cfg['meas_parameters']['n0_rng'][1],
                                  cfg['meas_parameters']['b_rng'][0],
                                  cfg['meas_parameters']['b_rng'][1]],
                       'pxsize': [cfg['meas_parameters']['n0_pnts'],
                                  cfg['meas_parameters']['b_pnts']]
                       }

    dv.new(stngs['file_name']+"-plot", ("i", "j", var_name1, var_name2), ('Cs', 'Ds', 'N', 'X', 'Y', 't'))
    print("Created {}".format(dv.get_name()))
    dv.add_comment(cfg['file']['comment'])
    dv.add_parameters(cfg['measurement'].items())
    dv.add_parameters(cfg['acbox_settings'].items())
    dv.add_parameters(cfg['lockin_settings'].items())
    dv.add_parameters(cfg['balancing_settings'].items()[:-1])  # check if it will handle the offset list
    dv.add_parameter('zoffet', cfg['meas_parameters']['zOffset'])
    dv.add_parameter('n0_rng', cfg['meas_parameters']['n0_rng'])
    dv.add_parameter('b_pnts', cfg['meas_parameters']['b_pnts'])
    dv.add_parameter('n0_pnts', cfg['meas_parameters']['n0_pnts'])
    dv.add_parameter('b_rng', cfg['meas_parameters']['b_rng'])
    dv.add_parameter('extent', tuple(plot_parameters['extent']))
    dv.add_parameter('pxsize', plot_parameters['pxsize'])

    if kwargs is not None:
        for key, value in kwargs.items():
            dv.add_parameter(key, value)

def rampfield(mag, field):
    print 'Ramping field to %1.5f ...' % field
    mag.conf_field_targ(field)
    mag.ramp()
    target_field = float(mag.get_field_targ())
    actual_field = float(mag.get_field_mag())
    while abs(target_field - actual_field) > 1e-4:
        time.sleep(1)
        actual_field = float(mag.get_field_mag())
    print 'Target field reached.'

def main():
    cxn = labrad.connect()
    dv = cxn.data_vault
    dc = cxn.dac_adc

    mag = cxn.ami_430
    mag.select_device()
    mag.conf_field_units(1)
    time.sleep(1)

    lck = cxn.amatek_7280_lock_in_amplifier
    lck.select_device()

    acbox = cxn.acbox
    quad_dc = cxn.dcbox_quad_ad5780;
    quad_dc.select_device()

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    ratio = init_acbox(acbox, cfg['acbox_settings'])

    dc.select_device()
    dc.set_conversiontime(int(read_ch1), ADC_CONVERSIONTIME)
    dc.set_conversiontime(int(read_ch2), ADC_CONVERSIONTIME)

    v_fixed = float(quad_dc.get_voltage(1)) - cfg['meas_parameters']['zOffset']  # this might need to have a case structure based on whats fixed?


    meas_parameters = cfg['meas_parameters']
    plot_parameters = cfg['plot_parameters']
    if meas_parameters['scale']:
        if meas_parameters['b_rng'][0] < meas_parameters['b_rng'][1]:
            bf = meas_parameters['b_rng']
            bscale = 1.0*bf[1]/bf[0]
            plot_parameters['pxsize'][0] *= bscale
            plot_parameters['extent'][0] *= bscale
            plot_parameters['extent'][1] *= bscale
            print("The data will be scaled and will take more time; new plot extent and pxsize: {}".format(plot_parameters))
    elif meas_parameters['b_rng'][0] == meas_parameters['b_rng'][1]:
            print("The start and final magnetic field match, Aligning?")
            plot_parameters['extent'][2] = plot_parameters['extent'][0]
            plot_parameters['extent'][3] = -1*plot_parameters['extent'][0]
            print("Plot extent is:{}".format(plot_parameters['extent']))


    cb = init_bridge(cxn, ratio, cfg['balancing_settings'])


    f = function_select(cfg['measurement']['fixed'])
    v1_balance = f(nbal, v_fixed)

    lck.tc(int(round(3 * math.log(balance_tc, 10) + 18)))
    dc.set_voltage(ch_x, v1_balance)

    print cb.balance(s1, s2)
    cs, ds = cb.get_physical_quantitites()
    print("Via balance: Cs = {}, Ds = {}".format(cs, ds))
    c_, d_ = cb.offbalance_scale()
    print("Scaling factors for offset: Ctg {} and Dtg {}".format(c_, d_))

    capacitance_params = {'Capacitance': cs, 'Dissipation': ds, 'offbalance c_': c_, 'offbalance d_': d_}


    create_file(dv, cfg, **{'vfixed': v_fixed, **capacitance_params}})

    ac_gain_var = int(round(gaindB / 6.666))
    tc_var = int(round(3 * math.log(tau, 10) + 18))
    sens_var = int(round(3 * math.log(sens, 10) + 27))
    lck.tc(tc_var)
    lck.set_ac_gain(ac_gain_var)
    lck.set_sensitivity(sens_var)

    s = lck.get_sensitivity()['mV']
    time.sleep(.25)
    t0 = time.time()

    num_x0, num_y = cfg['meas_parameters']['n0_pnts'], cfg['meas_parameters']['b_pnts']
    extent = tuple([x for t in (cfg['meas_parameters']['n0_rng'], cfg['meas_parameters']['b_rng']) for x in t])  # merge the extents
    pxsize = (num_x0, num_y)


    field = np.linspace(extent[2], extent[3], num_y)  # generate the array of fields

    est_time = (pxsize[0] * pxsize[1] + pxsize[1]) * DELAY_MEAS * 1e-6 / 60.0

    for i in range(num_y):
        rampfield(mag, field[i])

        if cfg['meas_parameters']['scale']:
            bscale = field[i]*1.0/field[0]  #calculates the scaling factor
        else:
            bscale = 1

        num_x = int(num_x0*bscale)  #scale the number of pixels
        ext_x = (extent[0]*bscale, extent[1]*bscale)  # scale the n0/c range

        mn = np.linspace(ext_x[0], ext_x[1], num_x)  # generate n0's
        vec_x = f(mn, v_fixed)  # generate the gate voltages
        vec_y = field[i]*np.ones(num_x)
        data_x = np.zeros(num_x)
        data_y = np.zeros(num_x)

        print bscale, num_x, ext_x


        mask = np.logical_and(vec_x <= X_MAX, vec_x >= X_MIN)
        if np.any(mask == True):
            start, stop = np.where(mask == True)[0][0], np.where(mask == True)[0][-1]

            num_points = stop - start + 1
            print("{} of {}  --> Ramping. Points: {}".format(i + 1, num_y, num_points))

            dc.buffer_ramp([int(var_ch1)], [int(read_ch1), int(read_ch2)],
                            [vec_x[start]],
                            [vec_x[stop]],
                            num_points, DELAY_MEAS, ADC_AVGSIZE)

            d_read = dc.serial_poll.future(2, num_points)
            d_tmp = d_read.result()

            data_x[start:stop + 1], data_y[start:stop + 1] = d_tmp
            data_x[start:stop + 1] = (data_x[start:stop + 1] / 2.5 * s - adc_offset[0]) / adc_slope[0]
            data_y[start:stop + 1] = (data_y[start:stop + 1] / 2.5 * s - adc_offset[1]) / adc_slope[1]

            d_cap = (c_ * data_x + d_ * data_y) + cs
            d_dis = (d_ * data_x - c_ * data_y) + ds

            j = np.linspace(0, num_x - 1, num_x)
            ii = np.ones(num_x) * i
            t1 = np.ones(num_x) * time.time() - t0
            totdata = np.array([j, ii, vec_x, vec_y, d_cap, d_dis, mn, data_x, data_y, t1])
            dv.add(totdata.T)

    print("it took {} s. to write data".format(time.time() - t0))


if __name__ == '__main__':
    main()
