import numpy as np
from frqi import FRQI
from qiskit import execute
from qiskit.providers.aer.backends import AerSimulator
from skimage import data
from skimage.transform import resize


class TestFRQI:

    GATE_SET = {"h", "x", "measure", "barrier", "ccry"}
    SHOTS = 8192
    BACKEND = AerSimulator()
    FRQI = FRQI()
    IMAGE1 = np.array([[0, 0], [0, 0]])
    IMAGE2 = np.array([[1, 1], [1, 1]])
    IMAGE3 = np.array([[0.5, 0.5], [0.5, 0.5]])
    ASTRONAUT = resize(data.astronaut(), (2, 2))

    def test_result_image1(self):

        expected_keys = ["0 00", "0 01", "0 10", "0 11"]
        qc = self.FRQI.image_quantum_circuit(image=self.IMAGE1, measurements=True)
        counts = (
            execute(experiments=qc, backend=self.BACKEND, shots=self.SHOTS)
            .result()
            .get_counts()
        )

        result = [int(key in expected_keys) for key, _ in counts.items()]
        true_list = np.ones(len(expected_keys))

        assert np.allclose(result, true_list)

    def test_result_image2(self):

        expected_keys = ["1 00", "1 01", "1 10", "1 11"]
        qc = self.FRQI.image_quantum_circuit(image=self.IMAGE2, measurements=True)
        counts = (
            execute(experiments=qc, backend=self.BACKEND, shots=self.SHOTS)
            .result()
            .get_counts()
        )

        result = [int(key in expected_keys) for key, _ in counts.items()]
        true_list = np.ones(len(expected_keys))

        assert np.allclose(result, true_list)

    def test_result_image3(self):

        expected_keys = ["0 00", "0 01", "0 10", "0 11", "1 00", "1 01", "1 10", "1 11"]
        qc = self.FRQI.image_quantum_circuit(image=self.IMAGE3, measurements=True)
        counts = (
            execute(experiments=qc, backend=self.BACKEND, shots=self.SHOTS)
            .result()
            .get_counts()
        )

        result = [int(key in expected_keys) for key, _ in counts.items()]
        true_list = np.ones(len(expected_keys))

        assert np.allclose(result, true_list)

    def test_result_rgb_image(self):

        expected_keys = [
            "0 0 0 00",
            "0 0 0 01",
            "0 0 0 10",
            "0 0 0 11",
            "0 0 1 00",
            "0 0 1 01",
            "0 0 1 10",
            "0 0 1 11",
            "0 1 0 00",
            "0 1 0 01",
            "0 1 0 10",
            "0 1 0 11",
            "0 1 1 00",
            "0 1 1 01",
            "0 1 1 10",
            "0 1 1 11",
            "1 0 0 00",
            "1 0 0 01",
            "1 0 0 10",
            "1 0 0 11",
            "1 0 1 00",
            "1 0 1 01",
            "1 0 1 10",
            "1 0 1 11",
            "1 1 0 00",
            "1 1 0 01",
            "1 1 0 10",
            "1 1 0 11",
            "1 1 1 00",
            "1 1 1 01",
            "1 1 1 10",
            "1 1 1 11",
        ]
        qc = self.FRQI.image_quantum_circuit(image=self.ASTRONAUT, measurements=True)
        counts = (
            execute(experiments=qc, backend=self.BACKEND, shots=self.SHOTS)
            .result()
            .get_counts()
        )
        result = [int(key in expected_keys) for key, _ in counts.items()]
        true_list = np.ones(len(expected_keys))

        assert np.allclose(result, true_list)

    def test_image_qc_gates(self):

        qc = self.FRQI.image_quantum_circuit(image=self.IMAGE1, measurements=True)
        qc_rgb = self.FRQI.image_quantum_circuit(
            image=self.ASTRONAUT, measurements=True
        )

        circuit_gates = list(qc.count_ops())
        result_test_gates = [gate in self.GATE_SET for gate in circuit_gates]
        circuit_gates_rgb = list(qc_rgb.count_ops())
        result_test_gates_rgb = [gate in self.GATE_SET for gate in circuit_gates_rgb]
        gates_true_list = np.ones(len(result_test_gates))

        assert np.allclose(result_test_gates, gates_true_list)
        assert np.allclose(result_test_gates_rgb, gates_true_list)

    def test_image_qc_gate_count(self):

        qc = self.FRQI.image_quantum_circuit(image=self.IMAGE1, measurements=True)
        qc_rgb = self.FRQI.image_quantum_circuit(
            image=self.ASTRONAUT, measurements=True
        )

        qc_gates_dict = dict(qc.count_ops())
        del qc_gates_dict["barrier"]
        qc_gates_dict_rgb = dict(qc_rgb.count_ops())
        del qc_gates_dict_rgb["barrier"]

        aux_bin_list = [
            bin(i)[2:] for i in range(self.IMAGE1.shape[0] * self.IMAGE1.shape[1])
        ]
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

        hadamard_count = qc.qregs[0].size
        measure_count = qc.cregs[0].size + qc.cregs[1].size
        mcry_count = 2 * qc.qregs[0].size
        x_count = np.array(
            [2 * str.count(bnumber, "0") for bnumber in binary_list]
        ).sum()

        qc_count_gate_list = [
            qc_gates_dict["h"],
            qc_gates_dict["measure"],
            qc_gates_dict["x"],
            qc_gates_dict["ccry"],
        ]
        qc_rgb_count_gate_list = [
            qc_gates_dict_rgb["h"],
            qc_gates_dict_rgb["measure"],
            qc_gates_dict_rgb["x"],
            qc_gates_dict_rgb["ccry"],
        ]
        count_gate_list = [hadamard_count, measure_count, x_count, mcry_count]
        count_gate_list_rgb = [
            hadamard_count,
            measure_count + 2,
            3 * x_count,
            3 * mcry_count,
        ]

        assert np.allclose(qc_count_gate_list, count_gate_list)
        assert np.allclose(qc_rgb_count_gate_list, count_gate_list_rgb)
