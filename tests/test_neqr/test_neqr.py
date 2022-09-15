import numpy as np
from neqr import NEQR
from qiskit import execute
from qiskit.providers.aer.backends import AerSimulator
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize


class TestNEQR:

    GATE_SET = {"ccx", "mcx", "h", "x", "measure", "barrier"}
    ASTRONAUT_IMAGE_GRAY = rgb2gray(data.astronaut())
    ZERO_IMAGE_MATRIX = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]])
    NEQR = NEQR()
    SHOTS = 8192
    BACKEND = AerSimulator()

    def _prepare_pixel_intensity_binary_dict(self, image_matrix: np.ndarray) -> dict:

        pixels_intensity = []
        for row in image_matrix:
            for entry in row:
                intensity = int(np.round(255 * entry))
                pixels_intensity.append(intensity)

        aux_binary_pixel_intensity = [
            bin(p_intensity)[2:] for p_intensity in pixels_intensity
        ]
        aux_len_bin_list = [
            len(binary_num) for binary_num in aux_binary_pixel_intensity
        ]
        max_length = max(aux_len_bin_list)
        binary_pixel_intensity = []

        for bnum in aux_binary_pixel_intensity:
            if len(bnum) < max_length:
                new_binary = ""
                for _ in range(max_length - len(bnum)):
                    new_binary += "0"
                new_binary += bnum
                binary_pixel_intensity.append(new_binary)
            else:
                binary_pixel_intensity.append(bnum)

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

        pixel_intensity_binary_dict = {
            tp[0]: tp[1] for tp in zip(binary_list, binary_pixel_intensity)
        }

        return pixel_intensity_binary_dict

    def _process_counts(self, counts: dict, image_matrix: np.ndarray) -> dict:

        num_pixels = image_matrix.shape[0] * image_matrix.shape[1]
        keys_list = [key for key, _ in sorted(counts.items())][:num_pixels]
        processed_counts = {key.split(" ")[0]: key.split(" ")[1] for key in keys_list}

        return processed_counts

    def test_image_qc_non_square_matrix_encoding(self):

        qc = self.NEQR.image_quantum_circuit(
            image=self.ZERO_IMAGE_MATRIX, measurements=True
        )

        counts = (
            execute(experiments=qc, backend=self.BACKEND, shots=self.SHOTS)
            .result()
            .get_counts()
        )

        processed_counts = self._process_counts(
            counts=counts, image_matrix=self.ZERO_IMAGE_MATRIX
        )
        pixel_intensity_dict = self._prepare_pixel_intensity_binary_dict(
            image_matrix=self.ZERO_IMAGE_MATRIX
        )

        results = [
            int(value == processed_counts[key])
            for key, value in pixel_intensity_dict.items()
        ]
        pixel_true_list = np.ones(len(results))

        assert np.allclose(results, pixel_true_list)

    def test_image_qc_non_square_matrix_gates(self):

        qc = self.NEQR.image_quantum_circuit(
            image=self.ZERO_IMAGE_MATRIX, measurements=True
        )

        circuit_gates = list(qc.count_ops())
        result_test_gates = [gate in self.GATE_SET for gate in circuit_gates]
        gates_true_list = np.ones(len(result_test_gates))

        assert np.allclose(result_test_gates, gates_true_list)

    def test_image_qc_non_square_matrix_gate_count(self):

        qc = self.NEQR.image_quantum_circuit(
            image=self.ZERO_IMAGE_MATRIX, measurements=True
        )

        qc_gates_dict = dict(qc.count_ops())
        del qc_gates_dict["barrier"]

        pixel_intensity_dict = self._prepare_pixel_intensity_binary_dict(
            image_matrix=self.ZERO_IMAGE_MATRIX
        )

        hadamard_count = qc.qregs[1].size
        measure_count = qc.cregs[0].size + qc.cregs[1].size
        x_count = np.array(
            [
                2 * str.count(key, "0")
                for key, value in pixel_intensity_dict.items()
                if value != "0" * 8
            ]
        ).sum()
        ccx_or_mcx_count = np.array(
            [str.count(bnum, "1") for _, bnum in pixel_intensity_dict.items()]
        ).sum()

        qc_count_gate_list = [
            qc_gates_dict["h"],
            qc_gates_dict["measure"],
            qc_gates_dict["x"],
            qc_gates_dict["mcx"],
        ]
        count_gate_list = [hadamard_count, measure_count, x_count, ccx_or_mcx_count]

        assert np.allclose(qc_count_gate_list, count_gate_list)

    def test_image_qc_square_matrix_encoding(self):

        resized_astronaut_pic = resize(self.ASTRONAUT_IMAGE_GRAY, (2, 2))
        qc = self.NEQR.image_quantum_circuit(
            image=resized_astronaut_pic, measurements=True
        )

        counts = (
            execute(experiments=qc, backend=self.BACKEND, shots=self.SHOTS)
            .result()
            .get_counts()
        )

        processed_counts = self._process_counts(
            counts=counts, image_matrix=resized_astronaut_pic
        )
        pixel_intensity_dict = self._prepare_pixel_intensity_binary_dict(
            image_matrix=resized_astronaut_pic
        )

        results = [
            int(value == processed_counts[key])
            for key, value in pixel_intensity_dict.items()
        ]
        pixel_true_list = np.ones(len(results))

        assert np.allclose(results, pixel_true_list)

    def test_image_qc_square_matrix_gates(self):

        resized_astronaut_pic = resize(self.ASTRONAUT_IMAGE_GRAY, (2, 2))
        qc = self.NEQR.image_quantum_circuit(
            image=resized_astronaut_pic, measurements=True
        )

        circuit_gates = list(qc.count_ops())
        result_test_gates = [gate in self.GATE_SET for gate in circuit_gates]
        gates_true_list = np.ones(len(result_test_gates))

        assert np.allclose(result_test_gates, gates_true_list)

    def test_image_qc_square_matrix_gate_count(self):

        resized_astronaut_pic = resize(self.ASTRONAUT_IMAGE_GRAY, (2, 2))
        qc = self.NEQR.image_quantum_circuit(
            image=resized_astronaut_pic, measurements=True
        )

        qc_gates_dict = dict(qc.count_ops())
        del qc_gates_dict["barrier"]

        pixel_intensity_dict = self._prepare_pixel_intensity_binary_dict(
            image_matrix=resized_astronaut_pic
        )

        hadamard_count = qc.qregs[1].size
        measure_count = qc.cregs[0].size + qc.cregs[1].size
        x_count = np.array(
            [
                2 * str.count(key, "0")
                for key, value in pixel_intensity_dict.items()
                if value != "0" * 8
            ]
        ).sum()
        ccx_or_mcx_count = np.array(
            [str.count(bnum, "1") for _, bnum in pixel_intensity_dict.items()]
        ).sum()

        qc_count_gate_list = [
            qc_gates_dict["h"],
            qc_gates_dict["measure"],
            qc_gates_dict["x"],
            qc_gates_dict["ccx"],
        ]
        count_gate_list = [hadamard_count, measure_count, x_count, ccx_or_mcx_count]

        assert np.allclose(qc_count_gate_list, count_gate_list)
