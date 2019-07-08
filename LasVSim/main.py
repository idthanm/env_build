# coding=utf-8
import sys
from LasVSim import lasvsim

def check_import():
    print('import success')

if __name__ == "__main__":
    lasvsim.create_simulation()

    for i in range(1300):
        lasvsim.sim_step_internal()
    #lasvsim.load_scenario(path='C:\Users\Chason\Desktop\\3_LasVSim')

    # while lasvsim.sim_step():
    #     pass
    lasvsim.export_simulation_data('C:/Users/Chason/Desktop/test.csv')
    print('done')
