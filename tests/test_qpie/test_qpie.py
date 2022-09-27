import numpy as np
from qpie import QPIE
from qiskit.providers.aer.backends import AerSimulator
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize


class TestQPIE:

    GATE_SET = {"initialize", "measure"}
    ASTRONAUT_IMAGE_GRAY = rgb2gray(data.astronaut())
    IMAGE = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    QPIE = QPIE()
    BACKEND = AerSimulator(method="statevector")

    def test_image_quantum_circuit_gates(self):

        resized_astro_gray_pic = resize(self.ASTRONAUT_IMAGE_GRAY, (8, 8))
        qc_with_measurements = self.QPIE.image_quantum_circuit(
            image=resized_astro_gray_pic, measurements=True
        )
        qc_without_measurements = self.QPIE.image_quantum_circuit(
            image=resized_astro_gray_pic
        )

        circuit_gates1 = list(qc_with_measurements.count_ops())
        circuit_gates2 = list(qc_without_measurements.count_ops())

        result_test_gates1 = [gate in self.GATE_SET for gate in circuit_gates1]
        result_test_gates2 = [gate in self.GATE_SET for gate in circuit_gates2]
        true_list1 = np.ones(len(result_test_gates1))
        true_list2 = np.ones(len(result_test_gates2))

        assert np.allclose(result_test_gates1, true_list1)
        assert np.allclose(result_test_gates2, true_list2)

    def test_recover_image_from_statevector(self):

        resized_astro_gray_pic = resize(self.ASTRONAUT_IMAGE_GRAY, (8, 8))
        normalization_factor1 = np.sqrt(
            np.sum(np.sum(resized_astro_gray_pic**2, axis=1))
        )
        normalized_image1 = resized_astro_gray_pic / normalization_factor1
        qc1 = self.QPIE.image_quantum_circuit(image=resized_astro_gray_pic)
        qc1.save_state()
        sv1 = self.BACKEND.run(qc1).result().get_statevector()
        image1 = np.real(sv1).reshape(resized_astro_gray_pic.shape)

        normalization_factor2 = np.sqrt(np.sum(np.sum(self.IMAGE**2, axis=1)))
        normalized_image2 = self.IMAGE / normalization_factor2
        qc2 = self.QPIE.image_quantum_circuit(image=self.IMAGE)
        qc2.save_state()
        sv2 = self.BACKEND.run(qc2).result().get_statevector()
        image2 = np.real(sv2).reshape(self.IMAGE.shape)

        assert np.allclose(normalized_image1, image1)
        assert np.allclose(normalized_image2, image2)
