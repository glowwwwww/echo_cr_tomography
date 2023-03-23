import os
import time
import datetime
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, pulse, circuit, transpile, assemble, schedule
from qiskit.circuit import Gate
from qiskit.tools.monitor import job_monitor
from qiskit.pulse.library import GaussianSquare
from qiskit.providers.fake_provider import FakeAthens, FakeBoeblingenV2, FakeOpenPulse2Q


from qiskit import assemble
from qiskit.providers.aer import PulseSimulator
from qiskit.providers.aer.pulse import PulseSystemModel


class Variable_Quantum_Circuit():
    def __init__(self, backend, cbit, tbit, amp, rise_ratio, num_shots):
        """
        Args:
            backend : qiskit.provider object
            t: int, number of timestep 
            U: int, label to set the state of the control qubit
            M: int, label to set the effective measurement basis
            cbit: int, number of control qubit
            tbit: int, number of target qubit
            amp: complex number, absolute value under 1

        Methods:
            run: simulate the circuit with t, U, M
        """
        self.backend = backend
        self.cbit = cbit
        self.tbit = tbit
        self.amp = amp
        self.rise_ratio = rise_ratio
        self.num_shots = num_shots
        self._init_backend()

    def _init_backend(self):
        self.backend_model = PulseSystemModel.from_backend(self.backend)
        self.backend_sim = PulseSimulator()
        self.qubit_lo_freq = self.backend_model.hamiltonian.get_qubit_lo_from_drift()
        self.dt = self.backend.configuration().dt


    def _build_cr_pulse(self, n_t, rise_ratio=0.1):
        with pulse.build(backend=self.backend) as cr_sched:
            ctrl_chan_01 = pulse.control_channels(self.cbit, self.tbit)[0]
            pulse.play(GaussianSquare(int(n_t), self.amp, int(n_t*rise_ratio), risefall_sigma_ratio=1), ctrl_chan_01)
        return cr_sched
    
    def _build_circuit(self, cr_sched, U, M):
        cr_gate = circuit.Gate("cr", num_qubits=1, params=[])
        sim_circuit = QuantumCircuit(2,1)
        if U == 1:
            sim_circuit.x(self.cbit)
        sim_circuit.append(cr_gate, [self.cbit])
        sim_circuit.barrier()
        if M == 0:
            sim_circuit.sdg(self.tbit)
            sim_circuit.h(self.tbit)
        elif M == 1:
            sim_circuit.h(self.tbit)
        sim_circuit.barrier()
        sim_circuit.add_calibration(cr_gate, [self.cbit], cr_sched)
        sim_circuit.measure(self.tbit, 0)
        return sim_circuit

    def _build_circ_schedule(self, n_t, U, M):
        cr_schedule = self._build_cr_pulse(n_t)
        circuit = self._build_circuit(cr_schedule, U, M)
        sim_transpiled = transpile(circuit, self.backend)
        return schedule(sim_transpiled, self.backend)

    def run(self, n_t, U, M):
        """
        one experiment
        args
        t: int, number of timestep 
        U: int, label to set the state of the control qubit
        M: int, label to set the effective measurement basis
        return:
        
        """
        schedule = self._build_circ_schedule(n_t, U, M)
        sim_qobj = assemble(schedule,
                            backend = self.backend_sim,
                            qubit_lo_freq = self.qubit_lo_freq,
                            meas_level = 1, 
                            meas_return = 'avg',
                            shots = self.num_shots)
        result = self.backend_sim.run(sim_qobj, system_model=self.backend_model).result()
        # counts of the 1 in the simulation
        return 1-2*np.real(result.get_memory(0)[0])

"""
running test
"""
# q_circuit = Variable_Quantum_Circuit(FakeAthens(), cbit=0, tbit=1, amp=1, rise_ratio=0.1, num_shots=1024)
# print(q_circuit.run(50, 0, 1))
