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

        qc = self._initialize_circuit(image=image)
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
        for i in range(len(qc.qregs)):
            qc.measure(qubit=qc.qregs[i], cbit=qc.cregs[i])
            if i != len(qc.qregs) - 1:
                qc.barrier()

        return qc

    def _initialize_circuit(self, image: np.ndarray) -> QuantumCircuit:
        """Initialize the NEQR circuit.

        Args:
            image (np.ndarray): The input image.

        Returns:
            QuantumCircuit: The NEQR circuit initialized.
        """
        num_qubits = len(bin((image.shape[0] * image.shape[1] - 1))[2:])
        qubits_index = QuantumRegister(size=num_qubits, name="pixels_indexes")
        intensity = QuantumRegister(size=8, name="intensity")
        bits_index = ClassicalRegister(size=num_qubits, name="bits_pixels_indexes")
        bits_intensity = ClassicalRegister(size=8, name="bits_intensity")

        if len(image.shape) == 3:
            # red = QuantumRegister(size=1, name="red")
            # green = QuantumRegister(size=1, name="green")
            # blue = QuantumRegister(size=1, name="blue")
            # bit_red = ClassicalRegister(size=1, name="bit_red")
            # bit_green = ClassicalRegister(size=1, name="bit_green")
            # bit_blue = ClassicalRegister(size=1, name="bit_blue")
            rgb = QuantumRegister(size=2, name="rgb")
            rgb_bits = ClassicalRegister(size=2, name="bits_rgb")

            qc = QuantumCircuit(
                intensity, qubits_index, rgb, bits_intensity, bits_index, rgb_bits
            )
            qc.h(qubit=rgb)
        else:
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

        len_image_shape = len(image.shape)

        if len_image_shape == 2:
            n = 1
        else:
            n = len_image_shape

        num_pixels = 2 ** len(qc.qregs[1])
        aux_bin_list = [bin(i)[2:] for i in range(num_pixels)][
            : image.shape[0] * image.shape[1]
        ]
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

        for j in range(n):
            pixels_intensity = []
            if n == 1:
                pixels_matrix = image
            else:
                pixels_matrix = image[:, :, j]
            for row in pixels_matrix:
                for entry in row:
                    intensity = int(np.round(255 * entry))
                    pixels_intensity.append(intensity)

            binary_pixel_intensity = [
                bin(p_intensity)[2:] for p_intensity in pixels_intensity
            ]

            for k, bnum in enumerate(binary_list):

                if binary_pixel_intensity[k] != "0":
                    for idx, element in enumerate(bnum[::-1]):
                        if element == "0":
                            qc.x(qubit=qc.qregs[1][idx])
                    if n != 1:
                        if j == 0:
                            qc.x(qubit=qc.qregs[2])
                        elif j == 1:
                            qc.x(qubit=qc.qregs[2][1])
                        elif j == 2:
                            qc.x(qubit=qc.qregs[2][0])

                    for idx, element in enumerate(binary_pixel_intensity[k][::-1]):
                        if element == "1":
                            if n == 1:
                                qc.mct(
                                    control_qubits=qc.qregs[1],
                                    target_qubit=qc.qregs[0][idx],
                                )
                            else:
                                control_qubits_list = list(qc.qregs[1]) + list(
                                    qc.qregs[2]
                                )
                                qc.mct(
                                    control_qubits=control_qubits_list,
                                    target_qubit=qc.qregs[0][idx],
                                )

                    for idx, element in enumerate(bnum[::-1]):
                        if element == "0":
                            qc.x(qubit=qc.qregs[1][idx])

                    if n != 1:
                        if j == 0:
                            qc.x(qubit=qc.qregs[2])
                        elif j == 1:
                            qc.x(qubit=qc.qregs[2][1])
                        elif j == 2:
                            qc.x(qubit=qc.qregs[2][0])
                    qc.barrier()

        return qc
