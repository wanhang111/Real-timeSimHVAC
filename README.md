# Real-timeSimHVAC
Online real-time simulating HVAC based on the FMU created from modelica

#Simulation and optimization of the HVAC
This project shows how a HVAC system is simulated by Modelica, Trnsys, Energy+, and et al.

HVAC system Currently, a typical HVAC system, based on two water-cooled chillers, two cooling water pumps, two cooling towers, two primary chilled water pumps, two secondary chilled water pumps, one AHU, and one simple room, has been built.

Simulation platform Open Modelica

Control strategy
Chillers: On/Off control based on a critical cooling load
Cooling towers: PID control of the fan to keep the supply water temperature at or aroundd the setpoint
Pumps: For constant speed pumps, the number of running pumps equals to the number of running chillers; For variable speed pumps, the number of running pumps is controlled by the speed signal and the mass flowrate, and the speed is controlled by maintaining a fixed differential pressure between the outlet and inlet on the waterside of the AHU.
AHU: PI control of the valve to keep the supply air temperature at the setpoint, PI control of the supply air fan to keep the return air temperature and the setpoint (which equals to the indoor air temperature)
