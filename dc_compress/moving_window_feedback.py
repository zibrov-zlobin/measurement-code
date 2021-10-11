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


I_LI_string = 'he-3-cryostat GPIB Bus - GPIB0::10::INSTR'

V_LI_string = 'he-3-cryostat GPIB Bus - GPIB0::9::INSTR'

time_const_val = .01
DELAY_MEAS = 3.3 * time_const_val * 1e6

initial_vtg = 0.5
final_vtg = 5.2
steps_vtg = 130



window_size = .25
window_steps = 180
conv_thr = window_size/window_steps * 20


### something like 4seconds overhead per line rn + 5s for wait time

tg_port = 0
bg_port = 1

def main():
    cxn = labrad.connect()
    dv = cxn.data_vault

    dv.cd('Ben')
    dv.cd('DC_C082721')
    dc = cxn.dac_adc_24bits
    ke = cxn.keithley_2450
    ke.select_device()
    Vsample_linspace = np.linspace(4.0,2.0,61)
    for Vsample in Vsample_linspace:
        ke.set_source_voltage(Vsample)
        time.sleep(3)
        measured_vsample = ke.measure_voltage()
        print('Sample voltage changed to: '+str(measured_vsample))
        title = '101021_4probe_610_78_4K_MWF_VSamp='+str(Vsample)+'V'
        comment = 'Sample at variable voltage V, 4K, 10 nA excitation'

        dv.new(title,("i","j",'Bg','Tg'),('I_X','I_Y','V_X','V_Y','R4p','FB'))
        # dv.add_parameter('Tg_pnts',steps_vtg+1)
        # dv.add_parameter('Tg_rng',(initial_vtg,final_vtg))



        dv.add_parameter('TgLoop-Start',initial_vtg)
        dv.add_parameter('TgLoop-End',final_vtg)
        dv.add_parameter('TgLoop-Steps',steps_vtg)
        dv.add_parameter('live_plots',[('Tg','FB')])
        dv.add_parameter('Vsample_meas',measured_vsample)
        dv.add_parameter('window size', window_size)
        dv.add_parameter('window_steps', window_steps)
        dv.add_comment(comment)

        lck = cxn.sr830
        lck.select_device(I_LI_string)
        I_sensitivity = lck.sensitivity()['V']
        I_sensitivity = I_sensitivity* 1e-7 ## correction for ithaco
        lck.select_device(V_LI_string)
        V_sensitivity = lck.sensitivity()['V']


        dc.select_device('he_3_cryostat_serial_server (COM4)')

        ##ramps to starting v_tg point
        current_vtop = dc.read_voltage(tg_port)
        Ramp_DACADC(dc, tg_port, current_vtop, initial_vtg, .25, .1)

        print('intitial equilibration time... (3 sec)')
        time.sleep(3)
        ##wide,initial sweep

        ### Initial Line Parameters
        initial_vback = -5
        final_vback = -1.8
        step_num = 300

        print('starting initial line...')
        current_vback = dc.read_voltage(bg_port)
        Ramp_DACADC(dc, bg_port, current_vback, initial_vback, .25, .1)


        d_tmp = dc.buffer_ramp([bg_port],[0,1,2,3],[initial_vback],[final_vback],step_num+1,DELAY_MEAS)
        Ix,Iy,Vx,Vy = d_tmp


        Tg = np.ones(step_num+1) * initial_vtg
        Bg = np.linspace(initial_vback,final_vback,step_num+1)
        Ix = I_sensitivity * Ix/10
        Iy = I_sensitivity * Iy/10
        Vx = V_sensitivity  * Vx/10
        Vy = V_sensitivity  * Vy/10

        try:
            R4p = Vx / Ix / 1000 #*1e-7
        except:
            R4p = Vx
        initial_bg = Bg[np.argmax(R4p)]
        print('Initial Back gate: '+ str(initial_bg))


        tg_range = np.linspace(initial_vtg,final_vtg,steps_vtg+1)
        current_vtop = dc.read_voltage(tg_port)
        current_feedback_pt = initial_bg
        initial_vback = current_feedback_pt - window_size/2
        final_vback = current_feedback_pt + window_size/2
        total_counter = 0
        for counter in range(0,len(tg_range)):
            Ramp_DACADC(dc, tg_port, current_vtop, tg_range[counter], .25, .1)
            current_vtop = tg_range[counter]
            if counter % 50 == 0:
                print('Vtg: '+str(current_vtop))

            for sub_counter in range(0,1):

                current_vback = dc.read_voltage(bg_port)

                Ramp_DACADC(dc, bg_port, current_vback, initial_vback, .25, .1)

                #for equilibration purposes
                time.sleep(.1)

                d_tmp = dc.buffer_ramp([bg_port],[0,1,2,3],[initial_vback],[final_vback],window_steps+1,DELAY_MEAS)
                Ix,Iy,Vx,Vy = d_tmp


                Tg = np.ones(window_steps+1) * current_vtop
                Bg = np.linspace(initial_vback,final_vback,window_steps+1)
                
                i = np.linspace(0,window_steps,window_steps+1)
                j = np.ones(window_steps+1)*total_counter

                Ix = I_sensitivity * Ix/10
                Iy = I_sensitivity * Iy/10
                Vx = V_sensitivity  * Vx/10
                Vy = V_sensitivity  * Vy/10
                try:
                    R4p = Vx / Ix / 1000
                except:
                    R4p = Vx
                old_feedback = current_feedback_pt
                current_feedback_pt = Bg[np.argmax(R4p)]


                center_val = np.ones(window_steps + 1) * current_feedback_pt
                initial_vback = current_feedback_pt - window_size/2
                final_vback = current_feedback_pt+window_size/2

                totdata = np.array([i,j,Bg,Tg,Ix,Iy,Vx,Vy,R4p,center_val]).T
                dv.add(totdata)
                if counter % 50 == 0:
                    print('New Back Gate: '+ str(current_feedback_pt))
                if abs(current_feedback_pt - old_feedback) < conv_thr:
                    #print('converged')
                    break
                total_counter += 1

    print('Done')
if __name__ == '__main__':
    main()


