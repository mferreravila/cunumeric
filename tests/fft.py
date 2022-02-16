# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import scipy.signal as sig

import cunumeric as num

def test_1d():
    Z     = np.random.rand(1000000) + np.random.rand(1000000) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft(Z)
    out_num   = num.fft(Z_num)

    # fumble      = np.random.rand(10) + np.random.rand(10) * 1j
    # out_fumble  = np.fft.ifft(fumble) 
    # out_fumble2 = np.fft.irfft(fumble)
    # print(out_fumble)
    # print(out_fumble2)

    # fumble      = np.random.rand(10)
    # fumble2     = fumble + np.random.rand(10) * 0j
    # print(fumble)
    # print(fumble2)
    # out_fumble  = np.fft.ifft(fumble) 
    # out_fumble2 = np.fft.ifft(fumble2)
    # print(out_fumble)
    # print(out_fumble2)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)

    assert np.allclose(out, out_num)

def test_1d_inverse():
    Z     = np.random.rand(1000000) + np.random.rand(1000000) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifft(Z, norm='forward')
    out_num   = num.ifft(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)

    assert np.allclose(out, out_num)

def test_1d_fp32():
    Z     = np.random.rand(1000000).astype(np.float32) + np.random.rand(1000000).astype(np.float32) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft(Z)
    out_num   = num.fft(Z_num)

    l2 = (out - out_num) * np.conj(out-out_num)
    l2 = np.sqrt(np.sum(l2)/np.sum(out*np.conj(out)))

    assert l2 < 1e-6

def test_2d():
    Z     = np.random.rand(128, 1024) + np.random.rand(128, 1024) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft2(Z)
    out_num   = num.fft(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_2d_inverse():
    Z     = np.random.rand(128, 1024) + np.random.rand(128, 1024) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifft2(Z, norm='forward')
    out_num   = num.ifft(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)


def test_3d():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fftn(Z)
    out_num   = num.fft(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_inverse():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifftn(Z, norm='forward')
    out_num   = num.ifft(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

if __name__ == "__main__":
    test_1d()
    test_1d_inverse()
    test_1d_fp32()
    test_2d()
    test_2d_inverse()
    test_3d()
    test_3d_inverse()
