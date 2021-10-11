import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import time
import math
import numpy as np
import labrad
import labrad.units as U
from include import CapacitanceBridge
import yaml

def Ramp_DACADC(DACADC_Device, Port, StartingVoltage, EndVoltage, StepSize, Delay, c = None):
    try:
        if StartingVoltage != EndVoltage:
            Delay = int(Delay * 1000000) #Delay in DAC is in microsecond
            Numberofsteps = int(abs(StartingVoltage - EndVoltage) / StepSize)
            if Numberofsteps < 2:
                Numberofsteps = 2
            g = DACADC_Device.ramp1(Port, float(StartingVoltage), float(EndVoltage), Numberofsteps, Delay)
            return g
    except Exception as inst:
        print('Error:', inst, ' on line: ', sys.exc_info()[2].tb_lineno)

RAMP_SPEED = 1.0 #volts/sec
RAMP_WAIT = 0.000  # seconds
X_MAX = 10
X_MIN = -8
Y_MAX = 4.25
Y_MIN = 0
ADC_CONVERSIONTIME = 250
ADC_AVGSIZE = 1

title = '4probe_711_1413_4K_PIDFeedback_sweepup_4.75V'
comment = 'Sample at +4.75 V, SiBg at -20 V 300ms TC'

I_LI_string = 'he-3-cryostat GPIB Bus - GPIB0::10::INSTR'

FS_LI_string = 'he-3-cryostat GPIB Bus - GPIB0::9::INSTR'

time_const_val = .3
DELAY_MEAS = 3.9 * time_const_val * 1e6

initial_vtg = -.962
final_vtg = 6.5
steps_vtg = 7500

### something like .4 seconds overhead / pt right now.

tg_port = 0
bg_port = 1

def main():
    cxn = labrad.connect()
    dv = cxn.data_vault
    dc = cxn.dac_adc_24bits

    dv.cd('Ben')
    dv.cd('DC_C082621')
    tg_varname = 'TG'
    dv.new(title,(tg_varname,'i'),('Bg','LI_X','LI_Y','FS'))
    dv.add_comment(comment)

    lck = cxn.sr830
    lck.select_device(I_LI_string)
    LI_sensitivity = lck.sensitivity()['V']

    lck.select_device(FS_LI_string)
    FS_LI_sensitivity = lck.sensitivity()['V']
    dc.select_device('he_3_cryostat_serial_server (COM4)')

    ##ramps to starting v_tg point
    current_vtop = dc.read_voltage(tg_port)
    #Ramp_DACADC(dc, tg_port, current_vtop, initial_vtg, .25, .1)
    #time.sleep(DELAY_MEAS/1e6)
    accumulated_error = 0

    tg_range = np.linspace(initial_vtg,final_vtg,steps_vtg+1)
    current_vtop = dc.read_voltage(tg_port)
    dv.add_parameter('TG_pnts',steps_vtg)
    dv.add_parameter('TG_rng',(initial_vtg,final_vtg))

    dv.add_parameter('i_pnts',steps_vtg)
    dv.add_parameter('i_rng',(0,steps_vtg))

    dv.add_parameter('TGLoop-Start',initial_vtg)
    dv.add_parameter('TGLoop-End',final_vtg)
    dv.add_parameter('TGLoop-Steps',steps_vtg)
    dv.add_parameter('live_plots',(('TG','Bg'),('i','LI_X')))
    for counter in range(0,len(tg_range)):
        if counter % 300 == 0:
            accumulated_error = 0
        if counter > 1:
            Ramp_DACADC(dc, tg_port, current_vtop, tg_range[counter], .25, .1)
        current_vtop = tg_range[counter]

        time.sleep(DELAY_MEAS/1e6)
        current_bg = dc.read_dac(bg_port)
        mult = 10
        LIXvals = np.zeros(mult)
        LIYvals = np.zeros(mult)
        LIFSvals = np.zeros(mult)
        for i in range(0,mult):
            LIXvals[i] = dc.read_voltage(0)*LI_sensitivity/10
            LIYvals[i] = dc.read_voltage(1)*LI_sensitivity/10
            LIFSvals[i] = dc.read_voltage(2)*FS_LI_sensitivity/10
        val_x = np.mean(LIXvals)
        val_y = np.mean(LIYvals)
        val_FS = np.mean(LIFSvals)
        row = [current_vtop,counter,current_bg,val_x,val_y,val_FS]
        dv.add(row)
        error = val_x / (-.01)
        shift_bg = error* 1 + accumulated_error * .075/(counter%300+1) ### estimated slope
        accumulated_error += error
        print('Counter: '+str(counter))
        new_bg = current_bg - shift_bg
        print('top gate: '+ str(current_vtop))
        print(new_bg) 
        dc.set_voltage(bg_port,new_bg)

    # print('intitial equilibration time... (30 sec)')
    # time.sleep(30)
    # ##wide,initial sweep

    # print('starting initial line...')
    # initial_vback = -5
    # final_vback = -3
    # step_num = 250
    # current_vback = dc.read_voltage(bg_port)
    # Ramp_DACADC(dc, bg_port, current_vback, initial_vback, .25, .1)


    # d_tmp = dc.buffer_ramp([bg_port],[0,1,2,3],[initial_vback],[final_vback],step_num+1,DELAY_MEAS)
    # Ix,Iy,Vx,Vy = d_tmp


    # Tg = np.ones(step_num+1) * initial_vtg
    # Bg = np.linspace(initial_vback,final_vback,step_num+1)
    # Ix = I_sensitivity * Ix/10
    # Iy = I_sensitivity * Iy/10
    # Vx = V_sensitivity  * Vx/10
    # Vy = V_sensitivity  * Vy/10

    # R4p = Vx / Ix / 1000

    # initial_bg = Bg[np.argmax(R4p)]
    # print('Initial Back gate: '+ str(initial_bg))

    # window_size = .08
    # window_steps = 120
    # conv_thr = window_size/window_steps * 1.2

    # tg_range = np.linspace(initial_vtg,final_vtg,steps_vtg+1)
    # current_vtop = dc.read_voltage(tg_port)
    # current_feedback_pt = initial_bg
    # initial_vback = current_feedback_pt - window_size/2
    # final_vback = current_feedback_pt + window_size/2
    # total_counter = 0
    # for counter in range(0,len(tg_range)):
    #     Ramp_DACADC(dc, tg_port, current_vtop, tg_range[counter], .25, .1)
    #     current_vtop = tg_range[counter]
    #     print('Vtg: '+str(current_vtop))

    #     for sub_counter in range(0,6):

    #         current_vback = dc.read_voltage(bg_port)

    #         Ramp_DACADC(dc, bg_port, current_vback, initial_vback, .25, .1)

    #         #for equilibration purposes
    #         time.sleep(2)

    #         d_tmp = dc.buffer_ramp([bg_port],[0,1,2,3],[initial_vback],[final_vback],window_steps+1,DELAY_MEAS)
    #         Ix,Iy,Vx,Vy = d_tmp


    #         Tg = np.ones(window_steps+1) * current_vtop
    #         Bg = np.linspace(initial_vback,final_vback,window_steps+1)
            
    #         i = np.linspace(0,window_steps,window_steps+1)
    #         j = np.ones(window_steps+1)*total_counter

    #         Ix = I_sensitivity * Ix/10
    #         Iy = I_sensitivity * Iy/10
    #         Vx = V_sensitivity  * Vx/10
    #         Vy = V_sensitivity  * Vy/10

    #         R4p = Vx / Ix / 1000
    #         old_feedback = current_feedback_pt
    #         current_feedback_pt = Bg[np.argmax(R4p)]


    #         center_val = np.ones(window_steps + 1) * current_feedback_pt
    #         initial_vback = current_feedback_pt - window_size/2
    #         final_vback = current_feedback_pt+window_size/2

    #         totdata = np.array([i,j,Bg,Tg,Ix,Iy,Vx,Vy,R4p,center_val]).T
    #         dv.add(totdata)
    #         print('New Back Gate: '+ str(current_feedback_pt))
    #         if abs(current_feedback_pt - old_feedback) < conv_thr:
    #             print('converged')
    #             break
    #         total_counter += 1

    print('Done')
if __name__ == '__main__':
    main()


