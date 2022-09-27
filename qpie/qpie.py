from __future__ import annotations
import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.providers.aer.backends import AerSimulator


class QPIE:
    """QPIE class"""

    def __init__(self) -> QPIE:
        pass

    def _amplitude_encode(self, image: np.ndarray) -> list:
        """_summary_

        Args:
            image (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        normalization_factor = np.sqrt(np.sum(np.sum(image**2, axis=1)))
        normalized_image = image / normalization_factor
        normalized_image = normalized_image.reshape(
            (image.shape[0] * image.shape[1], 1)
        )
        normalized_image_list = [num[0] for num in normalized_image]

        return normalized_image_list

    def _initialize_circuit(
        self, image: np.ndarray, measurements: bool = False
    ) -> QuantumCircuit:
        """_summary_

        Args:
            image (np.ndarray): _description_
            measurements (bool, optional): _description_. Defaults to False.

        Returns:
            QuantumCircuit: _description_
        """

        num_qubits = np.ceil(np.log2(image.shape[0] * image.shape[1]))
        qubits = QuantumRegister(size=num_qubits, name="pixel")

        if measurements:

            bits = ClassicalRegister(size=num_qubits, name="bits_pixel")
            qc = QuantumCircuit(qubits, bits)
        else:
            qc = QuantumCircuit(qubits)

        return qc

    def image_quantum_circuit(
        self, image: np.ndarray, measurements: bool = False
    ) -> QuantumCircuit:
        """_summary_

        Args:
            image (np.ndarray): _description_
            measurements (bool, optional): _description_. Defaults to False.

        Returns:
            QuantumCircuit: _description_
        """

        normalized_img = self._amplitude_encode(image=image)
        qc = self._initialize_circuit(image=image, measurements=measurements)
        qc.initialize(normalized_img, qc.qregs[0])
        if measurements:
            qc.measure(qubit=qc.qregs[0], cbit=qc.cregs[0])

        return qc

    def recover_image_from_statevector(
        self, quantum_circuit: QuantumCircuit, image_shape: tuple
    ) -> np.ndarray:
        """_summary_

        Args:
            quantum_circuit (QuantumCircuit): _description_
            image_shape (tuple): _description_

        Returns:
            np.ndarray: _description_
        """

        backend = AerSimulator(method="statevector")
        quantum_circuit.save_state()
        statevec = backend.run(quantum_circuit).result().get_statevector()
        image = np.real(statevec).reshape(image_shape)

        return image
