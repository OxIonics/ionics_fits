import numpy as np

from ionics_fits.models.transformations.scaled_model import ScaledModel
from ionics_fits.models.molmer_sorensen import MolmerSorensenFreq
from ionics_fits.models.sinusoid import SineMinMax
from ionics_fits.binomial import BinomialFitter


def test_160_ms(plot_failures):
    """Test for abnormal termination errors when fitting a MS scan"""

    delta = np.array(
        [
            -120000.0,
            120000.0,
            0.0,
            -60000.0,
            60000.0,
            90000.0,
            30000.0,
            -90000.0,
            -30000.0,
            -105000.0,
            -75000.0,
            45000.0,
            75000.0,
            -15000.0,
            -45000.0,
            105000.0,
            15000.0,
            -52500.0,
            82500.0,
            -97500.0,
            97500.0,
            37500.0,
            22500.0,
            112500.0,
            7500.0,
            52500.0,
            67500.0,
            -82500.0,
            -112500.0,
            -37500.0,
            -67500.0,
            -7500.0,
            -22500.0,
            -26250.0,
            -71250.0,
            -18750.0,
            101250.0,
            33750.0,
            3750.0,
            48750.0,
            -116250.0,
            93750.0,
            -56250.0,
            41250.0,
            63750.0,
            -48750.0,
            78750.0,
            11250.0,
            -33750.0,
            -3750.0,
            108750.0,
            -78750.0,
            -11250.0,
            -108750.0,
            116250.0,
            56250.0,
            86250.0,
            -41250.0,
            -86250.0,
            18750.0,
            71250.0,
            -101250.0,
            -63750.0,
            26250.0,
            -93750.0,
            -88125.0,
            -54375.0,
            -114375.0,
            50625.0,
            39375.0,
            -65625.0,
            -110625.0,
            13125.0,
            -28125.0,
            -20625.0,
            -58125.0,
            -9375.0,
            -95625.0,
            -76875.0,
            -73125.0,
            80625.0,
            46875.0,
            20625.0,
            24375.0,
            31875.0,
            16875.0,
            -69375.0,
            -103125.0,
            -5625.0,
            -61875.0,
            118125.0,
            -84375.0,
            -35625.0,
            61875.0,
            106875.0,
            54375.0,
            -43125.0,
            58125.0,
            43125.0,
            9375.0,
            73125.0,
            84375.0,
            -39375.0,
            -31875.0,
            114375.0,
            -50625.0,
            -91875.0,
            99375.0,
            -16875.0,
            65625.0,
            76875.0,
            -24375.0,
            69375.0,
            -46875.0,
            95625.0,
            -106875.0,
            88125.0,
            -118125.0,
            -99375.0,
            28125.0,
            91875.0,
            1875.0,
            110625.0,
            5625.0,
            -1875.0,
            -13125.0,
            103125.0,
            -80625.0,
            35625.0,
            51562.5,
            27187.5,
            45937.5,
            -115312.5,
            98437.5,
            -47812.5,
            -100312.5,
            -6562.5,
            -937.5,
            -87187.5,
            -81562.5,
            -105937.5,
            -17812.5,
            -89062.5,
            40312.5,
            -90937.5,
            100312.5,
            74062.5,
            107812.5,
            32812.5,
            -68437.5,
            -42187.5,
            -62812.5,
            115312.5,
            -55312.5,
            -49687.5,
            -45937.5,
            17812.5,
            -19687.5,
            34687.5,
            -57187.5,
            -4687.5,
            -36562.5,
            10312.5,
            -83437.5,
            59062.5,
            111562.5,
            -92812.5,
            -85312.5,
            -79687.5,
            -113437.5,
            -2812.5,
            29062.5,
            47812.5,
            -77812.5,
            30937.5,
            105937.5,
            -119062.5,
            -10312.5,
            -64687.5,
            -32812.5,
            79687.5,
            6562.5,
            -8437.5,
            -102187.5,
            -96562.5,
            55312.5,
            68437.5,
            -14062.5,
            -72187.5,
            49687.5,
            -111562.5,
            25312.5,
            -94687.5,
            87187.5,
            77812.5,
            -53437.5,
            109687.5,
            -66562.5,
            42187.5,
            44062.5,
            85312.5,
            -107812.5,
            -12187.5,
            -40312.5,
            2812.5,
            36562.5,
            -60937.5,
            64687.5,
            104062.5,
            -75937.5,
            62812.5,
            81562.5,
            70312.5,
            -70312.5,
            -109687.5,
            38437.5,
            19687.5,
            12187.5,
            15937.5,
            72187.5,
            -27187.5,
            92812.5,
            -104062.5,
            -15937.5,
            90937.5,
            60937.5,
            8437.5,
            -38437.5,
            -98437.5,
            75937.5,
            94687.5,
            57187.5,
            23437.5,
            119062.5,
            -29062.5,
            83437.5,
            14062.5,
            -21562.5,
            -59062.5,
            -117187.5,
            96562.5,
            4687.5,
            -51562.5,
            -34687.5,
            -25312.5,
            53437.5,
            117187.5,
            89062.5,
            66562.5,
            -30937.5,
            -44062.5,
            21562.5,
            -23437.5,
            937.5,
            -74062.5,
            113437.5,
            102187.5,
            61406.25,
            -95156.25,
            -38906.25,
            10781.25,
            -66093.75,
            8906.25,
            -12656.25,
            41718.75,
            119531.25,
            84843.75,
            -22031.25,
            -55781.25,
            107343.75,
            54843.75,
            -2343.75,
            -52031.25,
            6093.75,
            -103593.75,
            5156.25,
            70781.25,
        ]
    )

    P = np.array(
        [
            1.0,
            1.0,
            0.99,
            0.91,
            0.95,
            1.0,
            0.92,
            0.99,
            0.87,
            1.0,
            0.95,
            0.84,
            0.93,
            0.97,
            0.87,
            1.0,
            0.91,
            1.0,
            0.98,
            0.99,
            0.99,
            0.98,
            0.79,
            0.99,
            0.44,
            1.0,
            1.0,
            1.0,
            0.99,
            0.96,
            1.0,
            0.41,
            0.75,
            0.68,
            1.0,
            1.0,
            1.0,
            1.0,
            0.47,
            0.96,
            1.0,
            0.99,
            0.97,
            0.86,
            0.9,
            0.98,
            0.88,
            0.55,
            1.0,
            0.5,
            0.99,
            0.95,
            0.5,
            1.0,
            1.0,
            0.96,
            1.0,
            0.91,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.64,
            0.97,
            1.0,
            1.0,
            1.0,
            1.0,
            0.92,
            1.0,
            0.99,
            0.69,
            0.78,
            0.87,
            0.85,
            0.47,
            1.0,
            0.94,
            1.0,
            0.98,
            1.0,
            0.93,
            0.69,
            0.99,
            1.0,
            1.0,
            0.99,
            0.55,
            0.85,
            1.0,
            1.0,
            1.0,
            0.94,
            1.0,
            0.98,
            0.85,
            0.95,
            0.86,
            0.51,
            0.97,
            1.0,
            0.94,
            0.99,
            0.97,
            1.0,
            1.0,
            1.0,
            1.0,
            0.99,
            0.94,
            0.65,
            1.0,
            0.92,
            0.96,
            1.0,
            1.0,
            1.0,
            1.0,
            0.77,
            0.97,
            0.53,
            0.98,
            0.49,
            0.57,
            0.7,
            1.0,
            0.98,
            1.0,
            1.0,
            0.75,
            0.89,
            0.99,
            0.97,
            0.96,
            1.0,
            0.51,
            0.96,
            1.0,
            0.97,
            1.0,
            1.0,
            1.0,
            0.87,
            0.99,
            1.0,
            0.97,
            0.98,
            1.0,
            1.0,
            0.84,
            0.93,
            0.98,
            0.98,
            1.0,
            0.9,
            1.0,
            0.97,
            1.0,
            0.97,
            0.51,
            1.0,
            0.56,
            1.0,
            0.86,
            0.99,
            0.99,
            1.0,
            0.95,
            0.96,
            0.55,
            0.83,
            0.98,
            0.97,
            0.95,
            1.0,
            1.0,
            0.53,
            0.97,
            1.0,
            0.97,
            0.49,
            0.54,
            1.0,
            0.99,
            0.98,
            1.0,
            0.79,
            1.0,
            1.0,
            1.0,
            0.67,
            0.98,
            0.99,
            0.93,
            1.0,
            0.99,
            1.0,
            0.78,
            0.81,
            1.0,
            1.0,
            0.57,
            0.97,
            0.49,
            1.0,
            0.82,
            0.98,
            1.0,
            0.96,
            0.95,
            0.99,
            1.0,
            1.0,
            1.0,
            0.95,
            0.94,
            0.6,
            1.0,
            1.0,
            0.7,
            0.95,
            0.99,
            1.0,
            0.97,
            0.94,
            0.45,
            0.94,
            0.99,
            0.97,
            0.94,
            0.93,
            0.61,
            1.0,
            0.82,
            1.0,
            0.9,
            0.86,
            0.97,
            1.0,
            0.95,
            0.5,
            1.0,
            1.0,
            0.68,
            0.99,
            0.99,
            1.0,
            1.0,
            0.89,
            0.86,
            0.78,
            0.67,
            0.62,
            0.96,
            0.99,
            1.0,
            0.86,
            0.99,
            0.96,
            0.48,
            1.0,
            0.47,
            0.59,
            0.81,
            1.0,
            1.0,
            0.85,
            0.99,
            0.99,
            1.0,
            0.49,
            1.0,
            0.39,
            0.99,
            0.49,
            0.99,
        ]
    )

    inds = np.argsort(delta)
    delta = delta[inds]
    P = P[inds]

    num_shots = 100

    duration_estimate = 106.5e-6
    base_model = MolmerSorensenFreq(num_qubits=1, start_excited=True, walsh_idx=1)
    model = ScaledModel(model=base_model, x_scale=2 * np.pi)
    model.parameters["t_pulse"].user_estimate = duration_estimate
    model.parameters["omega"].user_estimate = np.pi / duration_estimate * np.sqrt(2)
    model.parameters["w_0"].fixed_to = 0.0
    BinomialFitter(delta, P, num_shots, model)


def test_160_sine():
    """
    Test for abnormal termination errors when fitting a sinusoid using the
    BinomialFitter.
    """
    phi = np.array(
        [
            0.0,
            0.075,
            0.09166667,
            0.10833333,
            0.125,
            0.14166667,
            0.15833333,
            0.175,
            0.25,
            0.325,
            0.34166667,
            0.35833333,
            0.375,
            0.39166667,
            0.40833333,
            0.425,
        ]
    )
    y = np.array(
        [
            4.99599452e-01,
            9.57476631e-02,
            4.34475939e-02,
            1.10580114e-02,
            1.47653008e-06,
            1.07666059e-02,
            4.28868985e-02,
            9.49598155e-02,
            4.98602728e-01,
            9.02244618e-01,
            9.54317348e-01,
            9.86437667e-01,
            9.97202809e-01,
            9.86146062e-01,
            9.53756262e-01,
            9.01456206e-01,
        ]
    )

    model = SineMinMax()
    model.parameters["min"].lower_bound = 0
    model.parameters["min"].upper_bound = 0.5
    model.parameters["max"].lower_bound = 0.5
    model.parameters["max"].upper_bound = 1

    BinomialFitter(x=phi, y=y, model=model, num_trials=1000)