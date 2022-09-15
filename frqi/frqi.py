from __future__ import annotations
import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import RYGate


class FRQI:
    """FRQI class"""

    def __init__(self) -> FRQI:
        pass

    def image_quantum_circuit(
        self, image: np.ndarray, measurements: bool = False
    ) -> QuantumCircuit:
        """Return a FRQI circuit that encodes the image given as input.

        Args:
            image (np.ndarray): The image that will be encoded.
            measurements (bool, optional): If we want to add measurements in the circuit.
                                           Defaults to False.

        Returns:
            QuantumCircuit: The FRQI circuit of the input image.
        """

        num_qubits = len(bin(image.shape[0] * image.shape[1] - 1)[2:])

        qc = self._initialize_circuit(num_qubits=num_qubits)
        qc = self._encode_image(quantum_circuit=qc, image=image)
        if measurements:
            qc = self._add_measurements(quantum_circuit=qc)

        return qc

    def _add_measurements(self, quantum_circuit: QuantumCircuit) -> QuantumCircuit:
        """Add measurements in FRQI circuit.

        Args:
            quantum_circuit (QuantumCircuit): A FRQI circuit that we want to
                                              add measurements.

        Returns:
            QuantumCircuit: FRQI circuit with measurements.
        """

        qc = quantum_circuit
        for i in range(len(qc.qregs)):
            qc.measure(qubit=qc.qregs[i], cbit=qc.cregs[i])
            if i != len(qc.qregs) - 1:
                qc.barrier()

        return qc

    def _initialize_circuit(self, num_qubits: int) -> QuantumCircuit:
        """Initialize the FRQI circuit.

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            QuantumCircuit: The FRQI circuit initialized.
        """

        pixels = QuantumRegister(size=num_qubits, name="pixels_indexes")
        intensity = QuantumRegister(size=1, name="intensity")
        bits = ClassicalRegister(size=num_qubits, name="bits_pixels_indexes")
        intensity_bit = ClassicalRegister(size=1, name="intensity_bit")

        qc = QuantumCircuit(pixels, intensity, bits, intensity_bit)

        qc.h(qubit=pixels)
        qc.barrier()

        return qc

    def _encode_image(
        self, quantum_circuit: QuantumCircuit, image: np.ndarray
    ) -> QuantumCircuit:
        """Encode an image in the quantum circuit.

        Args:
            quantum_circuit (QuantumCircuit): The initialized FRQI circuit.
            image (np.ndarray): The image that will be encoded
                                in the quantum circuit.

        Returns:
            QuantumCircuit: A full FRQI circuit.
        """

        qc = quantum_circuit
        qargs = list(qc.qregs[0]) + list(qc.qregs[1])

        pixels_intensity = []
        for row in image:
            for entry in row:
                intensity = (((entry * 255 * 3) / 17) / 90) * np.pi
                pixels_intensity.append(intensity)

        aux_bin_list = [bin(i)[2:] for i in range(len(pixels_intensity))]
        aux_len_bin_list = [len(binary) for binary in aux_bin_list]
        max_length = max(aux_len_bin_list)
        binary_list = []

        for bnum in aux_bin_list:
            if len(bnum) < max_length:
                new_binary = ""
                for _ in range(max_length - len(bnum)):
                    new_binary += "0"
                new_binary += bnum
                binary_list.append(new_binary)
            else:
                binary_list.append(bnum)

        for i, bnum in enumerate(binary_list):

            for idx, element in enumerate(bnum[::-1]):
                if element == "0":
                    qc.x(qubit=qc.qregs[0][idx])

            mcry = RYGate(theta=2 * pixels_intensity[i]).control(
                num_ctrl_qubits=len(qc.qregs[0])
            )
            qc.append(mcry, qargs=qargs)

            for idx, element in enumerate(bnum[::-1]):
                if element == "0":
                    qc.x(qubit=qc.qregs[0][idx])
            qc.barrier()

        return qc
