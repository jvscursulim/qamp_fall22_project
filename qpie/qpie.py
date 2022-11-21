from __future__ import annotations
import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.providers.aer.backends import AerSimulator


class QPIE:
    """QPIE class"""

    def __init__(self) -> QPIE:
        pass

    def _amplitude_encode(self, image: np.ndarray) -> list:
        """Return a list that represents the normalized image
        for amplitude encoding.

        Args:
            image (np.ndarray): The image that will be encoded.

        Returns:
            list: Return a list with the elements of the normalized input image.
        """

        normalization_factor = np.sqrt(np.sum(np.sum(image**2, axis=1)))
        normalized_image = image / normalization_factor
        num_elements = np.prod(image.shape)
        normalized_image = normalized_image.reshape(num_elements)
        normalized_image_list = [num for num in normalized_image]

        return normalized_image_list

    def image_quantum_circuit(
        self, image: np.ndarray, measurements: bool = False
    ) -> QuantumCircuit:
        """Return a QPIE circuit that encodes the image given as input.

        Args:
            image (np.ndarray): The image that will be encoded.
            measurements (bool, optional): If we want to add measurements in the circuit.
                                           Defaults to False.

        Returns:
            QuantumCircuit: The QPIE circuit of the input image.
        """

        normalized_img = self._amplitude_encode(image=image)
        num_elements = np.prod(image.shape)
        num_qubits = np.ceil(np.log2(num_elements))

        qubits = QuantumRegister(size=num_qubits, name="pixel")
        if measurements:

            bits = ClassicalRegister(size=num_qubits, name="bits_pixel")
            qc = QuantumCircuit(qubits, bits)
        else:
            qc = QuantumCircuit(qubits)

        qc.initialize(normalized_img, qc.qregs[0])
        if measurements:
            qc.measure(qubit=qc.qregs[0], cbit=qc.cregs[0])

        return qc

    def recover_image_from_statevector(
        self, quantum_circuit: QuantumCircuit, image_shape: tuple
    ) -> np.ndarray:
        """Reconstruct the image encoded on QPIE circuit.

        Args:
            quantum_circuit (QuantumCircuit): The QPIE circuit that encodes
                                              the input image.
            image_shape (tuple): The shape of the image that
                                 we want to reconstruct.

        Returns:
            np.ndarray: The image reconstructed from the statevector.
        """

        backend = AerSimulator(method="statevector")
        quantum_circuit.save_state()
        statevec = backend.run(quantum_circuit).result().get_statevector()
        image = np.real(statevec).reshape(image_shape)

        return image
