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

def allclose_float(A, B):
    l2 = (A - B) * np.conj(A-B)
    l2 = np.sqrt(np.sum(l2)/np.sum(A*np.conj(A)))
    return l2 < 1e-6

def test_1d():
    Z     = np.random.rand(1000000) + np.random.rand(1000000) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft(Z)
    out_num   = num.fft(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)

    assert np.allclose(out, out_num)

def test_1d_norm():
    Z     = np.random.rand(10) + np.random.rand(10) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft(Z, norm='forward')
    out_num   = num.fft(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)

    assert np.allclose(out, out_num)

def test_1d_s():
    Z     = np.random.rand(10) + np.random.rand(10) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft(Z, n=5)
    out_num   = num.fft(Z_num, n=5)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert np.allclose(out, out_num)

def test_1d_larger_s():
    Z     = np.random.rand(100) + np.random.rand(100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft(Z, n=250)
    out_num   = num.fft(Z_num, n=250)

    assert np.allclose(out, out_num)

def test_1d_inverse():
    Z     = np.random.rand(1000000) + np.random.rand(1000000) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifft(Z, norm='forward')
    out_num   = num.ifft(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)

    assert np.allclose(out, out_num)

def test_1d_inverse_norm():
    Z     = np.random.rand(1000000) + np.random.rand(1000000) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifft(Z)
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

    assert allclose_float(out, out_num)

def test_2d():
    Z     = np.random.rand(128, 1024) + np.random.rand(128, 1024) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft2(Z)
    out_num   = num.fft2(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_2d_norm():
    Z     = np.random.rand(128, 1024) + np.random.rand(128, 1024) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft2(Z, norm='forward')
    out_num   = num.fft2(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_2d_s():
    Z     = np.random.rand(128, 1024) + np.random.rand(128, 1024) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft2(Z,  s=(64,512))
    out_num   = num.fft2(Z_num, s=(64,512))

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_2d_larger_s():
    Z     = np.random.rand(128, 1024) + np.random.rand(128, 1024) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fft2(Z,  s=(129,1030))
    out_num   = num.fft2(Z_num, s=(129,1030))

    assert num.allclose(out, out_num)

    out2       = np.fft.fft2(Z,  s=(29,1030))
    out2_num   = num.fft2(Z_num, s=(29,1030))

    assert num.allclose(out2, out2_num)

def test_2d_axes():
    Z     = np.random.rand(128, 256) + np.random.rand(128, 256) * 1j
    Z_num = num.array(Z)

    out0       = np.fft.fft2(Z, axes=[0])
    out0_num   = num.fft2(Z_num, axes=[0])
    assert num.allclose(out0, out0_num)

    out1       = np.fft.fft2(Z, axes=[1])
    out1_num   = num.fft2(Z_num, axes=[1])
    assert num.allclose(out1, out1_num)

    out_1       = np.fft.fft2(Z, axes=[-1])
    out_1_num   = num.fft2(Z_num, axes=[-1])
    assert num.allclose(out_1, out_1_num)

    out_2       = np.fft.fft2(Z, axes=[-2])
    out_2_num   = num.fft2(Z_num, axes=[-2])
    assert num.allclose(out_2, out_2_num)

    out3       = np.fft.fft2(Z, axes=[0, 1])
    out3_num   = num.fft2(Z_num, axes=[0, 1])
    assert num.allclose(out3, out3_num)

    out4       = np.fft.fft2(Z, axes=[1, 0])
    out4_num   = num.fft2(Z_num, axes=[1, 0])
    assert num.allclose(out4, out4_num)

    out5       = np.fft.fft2(Z, axes=[1, 0, 1])
    out5_num   = num.fft2(Z_num, axes=[1, 0, 1])
    assert num.allclose(out5, out5_num)

def test_2d_inverse():
    Z     = np.random.rand(128, 1024) + np.random.rand(128, 1024) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifft2(Z, norm='forward')
    out_num   = num.ifft2(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_2d_inverse_norm():
    Z     = np.random.rand(128, 1024) + np.random.rand(128, 1024) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifft2(Z)
    out_num   = num.ifft2(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fftn(Z)
    out_num   = num.fftn(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_norm():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fftn(Z, norm='forward')
    out_num   = num.fftn(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_norm_ortho():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fftn(Z, norm='ortho')
    out_num   = num.fftn(Z_num, norm='ortho')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_s():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fftn(Z, s=(63,20,99))
    out_num   = num.fftn(Z_num, s=(63,20,99))

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_larger_s():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.fftn(Z, s=(65,43,109))
    out_num   = num.fftn(Z_num, s=(65,43,109))

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_axes():
    Z     = np.random.rand(2, 10, 3) + np.random.rand(2, 10, 3) * 1j
    Z_num = num.array(Z)

    out0       = np.fft.fftn(Z, axes=[0])
    out0_num   = num.fftn(Z_num, axes=[0])
    assert num.allclose(out0, out0_num)

    out1       = np.fft.fftn(Z, axes=[1])
    out1_num   = num.fftn(Z_num, axes=[1])
    assert num.allclose(out1, out1_num)

    out2       = np.fft.fftn(Z, axes=[2])
    out2_num   = num.fftn(Z_num, axes=[2])
    assert num.allclose(out2, out2_num)

    out_minus1       = np.fft.fftn(Z, axes=[-1])
    out_minus1_num   = num.fftn(Z_num, axes=[-1])
    assert num.allclose(out_minus1, out_minus1_num)

    out_minus2       = np.fft.fftn(Z, axes=[-2])
    out_minus2_num   = num.fftn(Z_num, axes=[-2])
    assert num.allclose(out_minus2, out_minus2_num)

    out_minus5       = np.fft.fftn(Z, axes=[-3])
    out_minus5_num   = num.fftn(Z_num, axes=[-3])
    assert num.allclose(out_minus5, out_minus5_num)

    out3       = np.fft.fftn(Z, axes=[2, 1])
    out3_num   = num.fftn(Z_num, axes=[2, 1])
    assert num.allclose(out3, out3_num)

    out4       = np.fft.fftn(Z, axes=[0, 2])
    out4_num   = num.fftn(Z_num, axes=[0, 2])
    assert num.allclose(out4, out4_num)

    out5       = np.fft.fftn(Z, axes=[0, 2, 1, 1, -1])
    out5_num   = num.fftn(Z_num, axes=[0, 2, 1, 1, -1])
    assert num.allclose(Z, Z_num)
    assert num.allclose(out5, out5_num)

    # print(Z)
    # print('-----------------------------------------------------------------------')
    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)

def test_3d_inverse():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifftn(Z, norm='forward')
    out_num   = num.ifftn(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_inverse_norm():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifftn(Z)
    out_num   = num.ifftn(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_inverse_s():
    Z     = np.random.rand(64, 40, 100) + np.random.rand(64, 40, 100) * 1j
    Z_num = num.array(Z)

    out       = np.fft.ifftn(Z, s=(12,37,50), norm='forward')
    out_num   = num.ifftn(Z_num, s=(12,37,50),norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_1d_r2c():
    Z     = np.random.rand(1000001)
    Z_num = num.array(Z)

    out       = np.fft.rfft(Z)
    out_num   = num.rfft(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert np.allclose(out, out_num)

def test_1d_fp32_r2c():
    Z     = np.random.rand(10).astype(np.float32)
    Z_num = num.array(Z)

    out       = np.fft.rfft(Z)
    out_num   = num.rfft(Z_num)

    l2 = (out - out_num) * np.conj(out-out_num)
    l2 = np.sqrt(np.sum(l2)/np.sum(out*np.conj(out)))

    assert l2 < 1e-6

def test_1d_s_r2c():
    Z     = np.random.rand(101)
    Z_num = num.array(Z)

    out       = np.fft.rfft(Z, n=11)
    out_num   = num.rfft(Z_num, n=11)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert np.allclose(out, out_num)

def test_2d_r2c ():
    Z     = np.random.rand(4, 2)
    Z_num = num.array(Z)
    print('INPUT')
    print(Z)
    print()

    out       = np.fft.rfft2(Z)
    out_num   = num.rfft2(Z_num)

    print(out)
    print('-----------------------------------------------------------------------')
    print(out_num)
    assert num.allclose(out, out_num)

def test_2d_s_r2c():
    Z     = np.random.rand(128, 1025)
    Z_num = num.array(Z)

    out       = np.fft.rfft2(Z,  s=(68,513))
    out_num   = num.rfft2(Z_num, s=(68,513))

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_2d_axes_r2c():
    Z     = np.random.rand(43, 197)
    Z_num = num.array(Z)
    
    out0       = np.fft.rfft2(Z, axes=[0])
    out0_num   = num.rfft2(Z_num, axes=[0])
    assert num.allclose(out0, out0_num)

    out1       = np.fft.rfft2(Z, axes=[1])
    out1_num   = num.rfft2(Z_num, axes=[1])
    assert num.allclose(out1, out1_num)

    out3       = np.fft.rfft2(Z, axes=[0, 1])
    out3_num   = num.rfft2(Z_num, axes=[0, 1])
    assert num.allclose(out3, out3_num)

    out4       = np.fft.rfft2(Z, axes=[1, 0])
    out4_num   = num.rfft2(Z_num, axes=[1, 0])

    out_1       = np.fft.rfft2(Z, axes=[-1])
    out_1_num   = num.rfft2(Z_num, axes=[-1])
    assert num.allclose(out_1, out_1_num)

    out_2       = np.fft.rfft2(Z, axes=[-2])
    out_2_num   = num.rfft2(Z_num, axes=[-2])
    assert num.allclose(out_2, out_2_num)

    out5       = np.fft.rfft2(Z, axes=[1, 0, 1])
    out5_num   = num.rfft2(Z_num, axes=[1, 0, 1])
    assert num.allclose(out5, out5_num)


def test_3d_r2c():
    Z     = np.random.rand(65, 42, 101)
    Z_num = num.array(Z)

    out       = np.fft.rfftn(Z)
    out_num   = num.rfftn(Z_num)

    assert num.allclose(out, out_num)

def test_3d_r2c_norm():
    Z     = np.random.rand(65, 42, 101)
    Z_num = num.array(Z)

    out       = np.fft.rfftn(Z, norm='forward')
    out_num   = num.rfftn(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_s_r2c():
    Z     = np.random.rand(64, 40, 100)
    Z_num = num.array(Z)

    out       = np.fft.rfftn(Z, s=(63,21,99))
    out_num   = num.rfftn(Z_num, s=(63,21,99))

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_axes_r2c():
    Z     = np.random.rand(20, 13, 8)
    Z_num = num.array(Z)

    out0       = np.fft.rfftn(Z, axes=[0])
    out0_num   = num.rfftn(Z_num, axes=[0])
    assert num.allclose(out0, out0_num)

    out1       = np.fft.rfftn(Z, axes=[1])
    out1_num   = num.rfftn(Z_num, axes=[1])
    assert num.allclose(out1, out1_num)

    out2       = np.fft.rfftn(Z, axes=[2])
    out2_num   = num.rfftn(Z_num, axes=[2])
    assert num.allclose(out2, out2_num)

    out_minus1       = np.fft.rfftn(Z, axes=[-1])
    out_minus1_num   = num.rfftn(Z_num, axes=[-1])
    assert num.allclose(out_minus1, out_minus1_num)

    out_minus2       = np.fft.rfftn(Z, axes=[-2])
    out_minus2_num   = num.rfftn(Z_num, axes=[-2])
    assert num.allclose(out_minus2, out_minus2_num)

    out_minus5       = np.fft.rfftn(Z, axes=[-3])
    out_minus5_num   = num.rfftn(Z_num, axes=[-3])
    assert num.allclose(out_minus5, out_minus5_num)

    out3       = np.fft.rfftn(Z, axes=[2, 1])
    out3_num   = num.rfftn(Z_num, axes=[2, 1])
    assert num.allclose(out3, out3_num)

    out4       = np.fft.rfftn(Z, axes=[0, 2])
    out4_num   = num.rfftn(Z_num, axes=[0, 2])
    assert num.allclose(out4, out4_num)

    out5       = np.fft.rfftn(Z, axes=[0, 2, 1, 1, -1])
    out5_num   = num.rfftn(Z_num, axes=[0, 2, 1, 1, -1])
    assert num.allclose(Z, Z_num)
    assert num.allclose(out5, out5_num)

    Z_float = Z.astype(np.float32)
    Z_num_float = num.array(Z_float)

    float_out0       = np.fft.rfftn(Z_float, axes=[0])
    float_out0_num   = num.rfftn(Z_num_float, axes=[0])
    assert allclose_float(float_out0, float_out0_num)

    float_out1       = np.fft.rfftn(Z_float, axes=[1])
    float_out1_num   = num.rfftn(Z_num_float, axes=[1])
    assert allclose_float(float_out1, float_out1_num)

    float_out2       = np.fft.rfftn(Z_float, axes=[2])
    float_out2_num   = num.rfftn(Z_num_float, axes=[2])
    assert allclose_float(float_out2, float_out2_num)

    float_out_minus1       = np.fft.rfftn(Z_float, axes=[-1])
    float_out_minus1_num   = num.rfftn(Z_num_float, axes=[-1])
    assert allclose_float(float_out_minus1, float_out_minus1_num)

    float_out_minus2       = np.fft.rfftn(Z_float, axes=[-2])
    float_out_minus2_num   = num.rfftn(Z_num_float, axes=[-2])
    assert allclose_float(float_out_minus2, float_out_minus2_num)

    float_out_minus5       = np.fft.rfftn(Z_float, axes=[-3])
    float_out_minus5_num   = num.rfftn(Z_num_float, axes=[-3])
    assert allclose_float(float_out_minus5, float_out_minus5_num)

    float_out3       = np.fft.rfftn(Z_float, axes=[2, 1])
    float_out3_num   = num.rfftn(Z_num_float, axes=[2, 1])
    assert allclose_float(float_out3, float_out3_num)

    float_out4       = np.fft.rfftn(Z_float, axes=[0, 2])
    float_out4_num   = num.rfftn(Z_num_float, axes=[0, 2])
    assert allclose_float(float_out4, float_out4_num)

    float_out5       = np.fft.rfftn(Z_float, axes=[0, 2, 1, 1, -1])
    float_out5_num   = num.rfftn(Z_num_float, axes=[0, 2, 1, 1, -1])
    assert allclose_float(float_out5, float_out5_num)

    # print(Z)
    # print('-----------------------------------------------------------------------')
    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)

def test_1d_c2r():
    Z     = np.random.rand(1000) + np.random.rand(1000) * 1j
    Z_num = num.array(Z)

    out       = np.fft.irfft(Z, norm='forward')
    out_num   = num.irfft(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)

    assert np.allclose(out, out_num)

def test_2d_c2r():
    Z     = np.random.rand(128, 512) + np.random.rand(128, 512) * 1j
    Z_num = num.array(Z)

    out       = np.fft.irfft2(Z, norm='forward')
    out_num   = num.irfft2(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_2d_axes_c2r():
    Z     = np.random.rand(2, 3) + np.random.rand(2, 3) * 1j
    Z_num = num.array(Z)
    
    out0       = np.fft.irfft2(Z, axes=[0])
    out0_num   = num.irfft2(Z_num, axes=[0])
    assert num.allclose(out0, out0_num)

    out1       = np.fft.irfft2(Z, axes=[1])
    out1_num   = num.irfft2(Z_num, axes=[1])
    assert num.allclose(out1, out1_num)

    out3       = np.fft.irfft2(Z, axes=[0, 1])
    out3_num   = num.irfft2(Z_num, axes=[0, 1])
    assert num.allclose(out3, out3_num)

    out4       = np.fft.irfft2(Z, axes=[1, 0])
    out4_num   = num.irfft2(Z_num, axes=[1, 0])
    assert num.allclose(out4, out4_num)

    out_1       = np.fft.irfft2(Z, axes=[-1])
    out_1_num   = num.irfft2(Z_num, axes=[-1])
    assert num.allclose(out_1, out_1_num)

    out_2       = np.fft.irfft2(Z, axes=[-2])
    out_2_num   = num.irfft2(Z_num, axes=[-2])
    assert num.allclose(out_2, out_2_num)

    out5       = np.fft.irfft2(Z, axes=[1, 0, 1])
    out5_num   = num.irfft2(Z_num, axes=[1, 0, 1])
    assert num.allclose(out5, out5_num)

def test_3d_c2r():
    Z     = np.random.rand(64, 40, 50) + np.random.rand(64, 40, 50) * 1j
    Z_num = num.array(Z)

    out       = np.fft.irfftn(Z, norm='forward')
    out_num   = num.irfftn(Z_num, norm='forward')

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_c2r_norm():
    Z     = np.random.rand(64, 40, 50) + np.random.rand(64, 40, 50) * 1j
    Z_num = num.array(Z)

    out       = np.fft.irfftn(Z)
    out_num   = num.irfftn(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert num.allclose(out, out_num)

def test_3d_axes_c2r():
    Z     = np.random.rand(20, 13, 8) + np.random.rand(20, 13, 8) * 1j
    Z_num = num.array(Z)

    out0       = np.fft.irfftn(Z, axes=[0])
    out0_num   = num.irfftn(Z_num, axes=[0])
    assert num.allclose(out0, out0_num)

    out1       = np.fft.irfftn(Z, axes=[1])
    out1_num   = num.irfftn(Z_num, axes=[1])
    assert num.allclose(out1, out1_num)

    out2       = np.fft.irfftn(Z, axes=[2])
    out2_num   = num.irfftn(Z_num, axes=[2])
    assert num.allclose(out2, out2_num)

    out_minus1       = np.fft.irfftn(Z, axes=[-1])
    out_minus1_num   = num.irfftn(Z_num, axes=[-1])
    assert num.allclose(out_minus1, out_minus1_num)

    out_minus2       = np.fft.irfftn(Z, axes=[-2])
    out_minus2_num   = num.irfftn(Z_num, axes=[-2])
    assert num.allclose(out_minus2, out_minus2_num)

    out_minus5       = np.fft.irfftn(Z, axes=[-3])
    out_minus5_num   = num.irfftn(Z_num, axes=[-3])
    assert num.allclose(out_minus5, out_minus5_num)

    out3       = np.fft.irfftn(Z, axes=[2, 1])
    out3_num   = num.irfftn(Z_num, axes=[2, 1])
    assert num.allclose(out3, out3_num)

    out4       = np.fft.irfftn(Z, axes=[0, 2])
    out4_num   = num.irfftn(Z_num, axes=[0, 2])
    assert num.allclose(out4, out4_num)

    out5       = np.fft.irfftn(Z, axes=[0, 2, 1, 1, -1])
    out5_num   = num.irfftn(Z_num, axes=[0, 2, 1, 1, -1])
    assert num.allclose(Z, Z_num)
    assert num.allclose(out5, out5_num)

    Z_float = Z.astype(np.complex64)
    Z_num_float = num.array(Z_float)

    float_out0       = np.fft.irfftn(Z_float, axes=[0])
    float_out0_num   = num.irfftn(Z_num_float, axes=[0])
    assert allclose_float(float_out0, float_out0_num)

    float_out1       = np.fft.irfftn(Z_float, axes=[1])
    float_out1_num   = num.irfftn(Z_num_float, axes=[1])
    assert allclose_float(float_out1, float_out1_num)

    float_out2       = np.fft.irfftn(Z_float, axes=[2])
    float_out2_num   = num.irfftn(Z_num_float, axes=[2])
    assert allclose_float(float_out2, float_out2_num)

    float_out_minus1       = np.fft.irfftn(Z_float, axes=[-1])
    float_out_minus1_num   = num.irfftn(Z_num_float, axes=[-1])
    assert allclose_float(float_out_minus1, float_out_minus1_num)

    float_out_minus2       = np.fft.irfftn(Z_float, axes=[-2])
    float_out_minus2_num   = num.irfftn(Z_num_float, axes=[-2])
    assert allclose_float(float_out_minus2, float_out_minus2_num)

    float_out_minus5       = np.fft.irfftn(Z_float, axes=[-3])
    float_out_minus5_num   = num.irfftn(Z_num_float, axes=[-3])
    assert allclose_float(float_out_minus5, float_out_minus5_num)

    float_out3       = np.fft.irfftn(Z_float, axes=[2, 1])
    float_out3_num   = num.irfftn(Z_num_float, axes=[2, 1])
    assert allclose_float(float_out3, float_out3_num)

    float_out4       = np.fft.irfftn(Z_float, axes=[0, 2])
    float_out4_num   = num.irfftn(Z_num_float, axes=[0, 2])
    assert allclose_float(float_out4, float_out4_num)

    float_out5       = np.fft.irfftn(Z_float, axes=[0, 2, 1, 1, -1])
    float_out5_num   = num.irfftn(Z_num_float, axes=[0, 2, 1, 1, -1])
    assert allclose_float(float_out5, float_out5_num)

def test_1d_hfft():
    Z     = np.random.rand(1000) + np.random.rand(1000) * 1j
    Z_num = num.array(Z)

    out       = np.fft.hfft(Z)
    out_num   = num.hfft(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)

    assert np.allclose(out, out_num)

def test_1d_hfft_inverse():
    Z     = np.random.rand(10)
    Z_num = num.array(Z)

    out       = np.fft.ihfft(Z)
    out_num   = num.ihfft(Z_num)

    # print(out)
    # print('-----------------------------------------------------------------------')
    # print(out_num)
    assert np.allclose(out, out_num)

if __name__ == "__main__":
    np.random.seed(0)
    test_1d()
    test_1d_norm()
    test_1d_s()
    test_1d_larger_s()
    test_1d_inverse()
    test_1d_inverse_norm()
    test_1d_fp32()
    test_2d()
    test_2d_norm()
    test_2d_s()
    test_2d_larger_s()
    test_2d_axes()
    test_2d_inverse()
    test_2d_inverse_norm()
    test_3d()
    test_3d_norm()
    test_3d_norm_ortho()
    test_3d_s()
    test_3d_larger_s()
    test_3d_axes()
    test_3d_inverse()
    test_3d_inverse_norm()
    test_3d_inverse_s()
    test_1d_r2c()
    test_1d_fp32_r2c()
    test_1d_s_r2c()
    test_2d_r2c()
    test_2d_s_r2c()
    test_3d_r2c()
    test_3d_r2c_norm()
    test_3d_s_r2c()
    test_1d_c2r()
    test_2d_c2r()
    test_3d_c2r()
    test_3d_c2r_norm()
    test_1d_hfft()
    test_1d_hfft_inverse()

    print()
    print()
    print()
    print()
    test_2d_axes_r2c()

    print()
    print()
    print()
    print()
    test_3d_axes_r2c()

    test_2d_axes_c2r()
    test_3d_axes_c2r()