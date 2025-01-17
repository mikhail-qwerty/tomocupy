import unittest
import os
import numpy as np
import tifffile
import inspect
import h5py

prefix = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 782.5 --nsino-per-chunk 4'
cmd_dict = {
    f'{prefix} ': 28.307,
    f'{prefix} --reconstruction-algorithm lprec ': 27.992,
    f'{prefix} --reconstruction-algorithm linerec ': 28.341,
    f'{prefix} --dtype float16': 24.186,
    f'{prefix} --reconstruction-algorithm lprec --dtype float16': 24.050,
    f'{prefix} --reconstruction-algorithm linerec --dtype float16': 25.543,
    f'{prefix} --binning 1': 12.286,
    f'{prefix} --reconstruction-algorithm lprec --binning 1': 12.252,
    f'{prefix} --reconstruction-algorithm linerec --binning 1': 12.259,
    f'{prefix} --start-row 3 --end-row 15 --start-proj 200 --end-proj 700': 17.589,
    f'{prefix} --save-format h5': 28.307,
    f'{prefix} --nsino-per-chunk 2 --file-type double_fov': 15.552,
    f'{prefix} --nsino-per-chunk 2 --blocked-views [0.2,1]': 30.790,
    f'{prefix} --nsino-per-chunk 2 --blocked-views [[0.2,1],[2,3]]': 40.849,
    f'{prefix} --remove-stripe-method fw': 28.167,
    f'{prefix} --remove-stripe-method fw --dtype float16': 23.945,
    f'{prefix} --start-column 200 --end-column 1000': 18.248,
    f'{prefix} --start-column 200 --end-column 1000 --binning 1': 7.945,
    f'{prefix} --flat-linear True': 28.308,
    f'{prefix} --rotation-axis-auto auto --rotation-axis-method sift  --reconstruction-type full' : 28.305,
    f'{prefix} --rotation-axis-auto auto --rotation-axis-method vo --center-search-step 0.1 --nsino 0.5 --center-search-width 100 --reconstruction-type full' : 28.303,
    f'{prefix} --remove-stripe-method vo-all ': 27.993,
}


class SequentialTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        testcase_methods = list(testCaseClass.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names


class Tests(unittest.TestCase):

    def test_full_recon(self):
        for cmd in cmd_dict.items():
            os.system('rm -rf data_rec')
            print(f'TEST {inspect.stack()[0][3]}: {cmd[0]}')
            st = os.system(cmd[0])
            self.assertEqual(st, 0)
            ssum = 0
            try:
                with h5py.File('data_rec/test_data_rec.h5', 'r') as fid:
                    data = fid['exchange/data']
                    ssum = np.sum(np.linalg.norm(data[:], axis=(1, 2)))
            except:
                pass
            for k in range(24):
                try:
                    ssum += np.linalg.norm(tifffile.imread(
                        f'data_rec/test_data_rec/recon_{k:05}.tiff'))
                except:
                    pass
            self.assertAlmostEqual(ssum, cmd[1], places=1)


if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)
