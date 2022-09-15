from __future__ import annotations
import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister


class NEQR:
    """NEQR class"""

    def __init__(self) -> NEQR:
        pass

    def image_quantum_circuit(
        self, image: np.ndarray, measurements: bool = False
    ) -> QuantumCircuit:
        """Return a NEQR circuit that encodes the image given as input.

        Args:
            image (np.ndarray): The image that will be encoded.
            measurements (bool, optional): If we want to add measurements in the circuit.
                                           Defaults to False.

        Returns:
            QuantumCircuit: The NEQR circuit of the input image.
        """

        num_qubits = len(bin((image.shape[0] * image.shape[1] - 1))[2:])

        qc = self._initialize_circuit(num_qubits=num_qubits)
        qc = self._encode_image(quantum_circuit=qc, image=image)
        if measurements:
            qc = self._add_measurements(quantum_circuit=qc)

        return qc

    def _add_measurements(self, quantum_circuit: QuantumCircuit) -> QuantumCircuit:
        """Add measurements in NEQR circuit.

        Args:
            quantum_circuit (QuantumCircuit): A quantum circuit that we want to
                                              add measurements.

        Returns:
            QuantumCircuit: A quantum circuit with measurements.
        """

        qc = quantum_circuit
        qc.measure(qubit=qc.qregs[0], cbit=qc.cregs[0])
        qc.barrier()
        qc.measure(qubit=qc.qregs[1], cbit=qc.cregs[1])

        return qc

    def _initialize_circuit(self, num_qubits: int) -> QuantumCircuit:
        """Initialize the NEQR circuit.

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            QuantumCircuit: The NEQR circuit initialized.
        """

        qubits_index = QuantumRegister(size=num_qubits, name="pixels_indexes")
        intensity = QuantumRegister(size=8, name="intensity")
        bits_index = ClassicalRegister(size=num_qubits, name="bits_pixels_indexes")
        bits_intensity = ClassicalRegister(size=8, name="bits_intensity")

        qc = QuantumCircuit(intensity, qubits_index, bits_intensity, bits_index)

        qc.h(qubit=qubits_index)
        qc.barrier()

        return qc

    def _encode_image(
        self, quantum_circuit: QuantumCircuit, image: np.ndarray
    ) -> QuantumCircuit:
        """Encode an image in the quantum circuit.

        Args:
            quantum_circuit (QuantumCircuit): The initialized NEQR circuit.
            image (np.ndarray): The image that will be encoded
                                in the quantum circuit.

        Returns:
            QuantumCircuit: A full NEQR circuit.
        """

        qc = quantum_circuit

        pixels_intensity = []
        for row in image:
            for entry in row:
                intensity = int(np.round(255 * entry))
                pixels_intensity.append(intensity)

        binary_pixel_intensity = [
            bin(p_intensity)[2:] for p_intensity in pixels_intensity
        ]
        aux_bin_list = [bin(i)[2:] for i in range(len(binary_pixel_intensity))]
        aux_len_bin_list = [len(binary_num) for binary_num in aux_bin_list]
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

            if binary_pixel_intensity[i] != "0":
                for idx, element in enumerate(bnum[::-1]):
                    if element == "0":
                        qc.x(qubit=qc.qregs[1][idx])

                for idx, element in enumerate(binary_pixel_intensity[i][::-1]):
                    if element == "1":
                        qc.mct(
                            control_qubits=qc.qregs[1], target_qubit=qc.qregs[0][idx]
                        )

                for idx, element in enumerate(bnum[::-1]):
                    if element == "0":
                        qc.x(qubit=qc.qregs[1][idx])
                qc.barrier()

        return qc
