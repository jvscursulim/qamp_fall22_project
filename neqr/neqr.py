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
        intensity = QuantumRegister(size=8, name="intensity")
        bits_intensity = ClassicalRegister(size=8, name="bits_intensity")

        if len(image.shape) == 3:
            if image.shape[2] == 3:
                num_qubits = int(np.ceil(np.log2(image.shape[0] * image.shape[1])))
                qubits_index = QuantumRegister(size=num_qubits, name="pixel_indexes")
                bits_index = ClassicalRegister(
                    size=num_qubits, name="bits_pixel_indexes"
                )
                rgb = QuantumRegister(size=2, name="rgb")
                rgb_bits = ClassicalRegister(size=2, name="bits_rgb")

                qc = QuantumCircuit(
                    intensity, qubits_index, rgb, bits_intensity, bits_index, rgb_bits
                )
                qc.h(qubit=rgb)
            else:
                num_qubits = int(
                    np.ceil(np.log2(image.shape[0] * image.shape[1] * image.shape[2]))
                )
                qubits_index = QuantumRegister(size=num_qubits, name="pixel_indexes")
                bits_index = ClassicalRegister(
                    size=num_qubits, name="bits_pixel_indexes"
                )
                qc = QuantumCircuit(intensity, qubits_index, bits_intensity, bits_index)

        else:
            num_qubits = int(np.ceil(np.log2(image.shape[0] * image.shape[1])))
            qubits_index = QuantumRegister(size=num_qubits, name="pixel_indexes")
            bits_index = ClassicalRegister(size=num_qubits, name="bits_pixel_indexes")
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
            if image.shape[2] == 3:
                n = len_image_shape
            else:
                n = 1

        num_pixel = 2 ** len(qc.qregs[1])
        if len(image.shape) == 3 and n == 1:
            aux_bin_list = [bin(i)[2:] for i in range(num_pixel)][
                : image.shape[0] * image.shape[1] * image.shape[2]
            ]
        else:
            aux_bin_list = [bin(i)[2:] for i in range(num_pixel)][
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
            pixel_intensity = []
            if n == 1:
                pixels_matrix = image
            else:
                pixels_matrix = image[:, :, j]
            if len(image.shape) == 3 and n == 1:
                for row in pixels_matrix:
                    for column in row:
                        for entry in column:
                            intensity = int(np.round(255 * entry))
                            pixel_intensity.append(intensity)
            else:
                for row in pixels_matrix:
                    for entry in row:
                        intensity = int(np.round(255 * entry))
                        pixel_intensity.append(intensity)

            binary_pixel_intensity = [
                bin(p_intensity)[2:] for p_intensity in pixel_intensity
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

    def _calculate_pixel_intensity_from_intensity_string(
        self, intensity_strings: list
    ) -> list:
        """Calculate the intensity of a pixel from a pixel string.

        Args:
            intensity_strings (list): A list with the binary strings
                                      that represents the intensities
                                      of each pixel in the image.

        Returns:
            list: An array with the pixels intensity.
        """

        pixel_intensity = []

        for string in intensity_strings:
            intensity = 0
            for idx, char in enumerate(string):
                if char == "1":
                    intensity += 2 ** (7 - idx)
            intensity = intensity / 255
            pixel_intensity.append(intensity)

        return pixel_intensity

    def reconstruct_image_from_neqr_result(
        self, counts: dict, image_shape: tuple
    ) -> np.ndarray:
        """Reconstruct the image encoded on NEQR circuit.

        Args:
            counts (dict): The dictionary with the results
                           of the experiments with NEQR circuit.
            image_shape (tuple): The shape of the image that
                                 we want to reconstruct.

        Raises:
            ValueError: If image_shape is not a tuple
                        with length equal to 2 or 3.

        Returns:
            np.ndarray: Image matrix.
        """

        keys_list = sorted(list(counts.keys()))

        if len(keys_list[0].split(" ")) == 2:
            if image_shape[0] == image_shape[1]:
                intensity_strings = [key.split(" ")[1] for key in keys_list]
            else:
                intensity_strings = [key.split(" ")[1] for key in keys_list][
                    : image_shape[0] * image_shape[1]
                ]
        elif len(keys_list[0].split(" ")) == 3:
            processed_keys = [key for key in keys_list if key.split(" ")[0] != "11"]
            if image_shape[0] == image_shape[1]:
                intensity_strings = [key.split(" ")[2] for key in processed_keys]
            else:
                intensity_strings = [key.split(" ")[2] for key in processed_keys][
                    : image_shape[0] * image_shape[1]
                ]

        pixel_intensity = self._calculate_pixel_intensity_from_intensity_string(
            intensity_strings=intensity_strings
        )

        if len(image_shape) == 3:
            if image_shape[2] == 3:
                pixel_intensity_rgb = np.split(np.array(pixel_intensity), 3)
                image = np.zeros(image_shape)
                for i, channel in enumerate(pixel_intensity_rgb):
                    channel_np = np.array(channel).reshape(
                        (image.shape[0], image.shape[1])
                    )
                    for j, row in enumerate(channel_np):
                        for k, entry in enumerate(row):
                            image[j, k, i] = entry
                return image
            else:
                image = np.array(pixel_intensity).reshape(image_shape)
                return image
        elif len(image_shape) == 2:
            image = np.array(pixel_intensity).reshape(image_shape)
            return image
        else:
            raise ValueError(
                "Image shape should be a tuple of length 2 for images in gray scale or a tuple of length 3 for RGB images and 3D images!"
            )
