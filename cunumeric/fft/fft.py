# Copyright 2021-2022 NVIDIA Corporation
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

import numpy as np

from cunumeric.array import ndarray
from cunumeric.module import add_boilerplate
from cunumeric.config import FFTCode, FFTDirection, FFTNormalization


def _sanitize_user_axes(a, s, axes):
    if s is None:
        if axes is None:
            s = list(a.shape)
        else:
            s = [list(a.shape)[ax] for ax in axes]
    if axes is None:
        axes = list(range(-len(s), 0))
    return s, axes


def _real_to_complex_kind(fft_type):
    if fft_type == FFTCode.FFT_D2Z or fft_type == FFTCode.FFT_Z2D:
        return FFTCode.FFT_Z2Z
    elif fft_type == FFTCode.FFT_R2C or fft_type == FFTCode.FFT_C2R:
        return FFTCode.FFT_C2C
    return fft_type


@add_boilerplate("a")
def fft(a, n=None, axis=None, norm=None):
    s = (n,) if n is not None else None
    return fftn(a=a, s=s, axes=axis, norm=norm)


@add_boilerplate("a")
def fft2(a, s=None, axes=None, norm=None):
    return fftn(a=a, s=s, axes=axes, norm=norm)


@add_boilerplate("a")
def fftn(a, s=None, axes=None, norm=None):
    # Check for types, no conversions for now
    fft_type = None
    if a.dtype == np.complex128:
        fft_type = FFTCode.FFT_Z2Z
    elif a.dtype == np.complex64:
        fft_type = FFTCode.FFT_C2C
    elif a.dtype == np.float64:
        fft_type = FFTCode.FFT_D2Z
    elif a.dtype == np.float32:
        fft_type = FFTCode.FFT_R2C
    else:
        raise TypeError("FFT input not supported, missing a conversion")
    return a.fft(s=s, axes=axes, kind=fft_type, direction=FFTDirection.FORWARD, norm=norm)


@add_boilerplate("a")
def ifft(a, n=None, axis=None, norm=None):
    s = (n,) if n is not None else None
    return ifftn(a=a, s=s, axes=axis, norm=norm)


@add_boilerplate("a")
def ifft2(a, s=None, axes=None, norm=None):
    # Ensure 2D
    return ifftn(a=a, s=s, axes=axes, norm=norm)


@add_boilerplate("a")
def ifftn(a, s=None, axes=None, norm=None):
    # Check for types, no conversions for now
    fft_type = None
    if a.dtype == np.complex128:
        fft_type = FFTCode.FFT_Z2Z
    elif a.dtype == np.complex64:
        fft_type = FFTCode.FFT_C2C
    # These two cases need promotion to complex
    # elif a.dtype == np.float64:
    #     fft_type = FFTCode.FFT_Z2Z
    # elif a.dtype == np.float32:
    #     fft_type = FFTCode.FFT_C2C
    else:
        raise TypeError("FFT input not supported, missing a conversion")
    return a.fft(s=s, axes=axes, kind=fft_type, direction=FFTDirection.INVERSE, norm=norm)


@add_boilerplate("a")
def rfft(a, n=None, axis=None, norm=None):
    s = (n,) if n is not None else None
    axis = (axis,) if axis is not None else None
    return rfftn(a=a, s=s, axes=axis, norm=norm)


@add_boilerplate("a")
def rfft2(a, s=None, axes=None, norm=None):
    return rfftn(a=a, s=s, axes=axes, norm=norm)


@add_boilerplate("a")
def rfftn(a, s=None, axes=None, norm=None):
    # Check for types, no conversions for now
    fft_type = None
    if a.dtype == np.float64:
        fft_type = FFTCode.FFT_D2Z
    elif a.dtype == np.float32:
        fft_type = FFTCode.FFT_R2C
    else:
        raise TypeError("FFT input not supported, missing a conversion")

    s, axes = _sanitize_user_axes(a, s, axes)
    print('S {} AXES {} DIM {}'.format(s,axes, a.ndim))

    operate_by_axes = (len(axes) != len(set(axes))) or (len(axes) != a.ndim)
    if not operate_by_axes:
        operate_by_axes = axes != sorted(axes)
        
    # Operate by axes
    if operate_by_axes:
        r2c = a.fft(s=[s[-1]], axes=[axes[-1]], kind=fft_type, direction=FFTDirection.FORWARD, norm=norm)
        if len(axes) > 1:
            return r2c.fft(s=s[0:-1], axes=axes[0:-1], kind=_real_to_complex_kind(fft_type), direction=FFTDirection.FORWARD, norm=norm)
        else:
            return r2c
    # Operate as a single FFT
    else:
        return a.fft(s=s, axes=axes, kind=fft_type, direction=FFTDirection.FORWARD, norm=norm)


@add_boilerplate("a")
def irfft(a, n=None, axis=None, norm=None):
    s = (n,) if n is not None else None
    return irfftn(a=a, s=s, axes=axis, norm=norm)


@add_boilerplate("a")
def irfft2(a, s=None, axes=None, norm=None):
    return irfftn(a=a, s=s, axes=axes, norm=norm)


@add_boilerplate("a")
def irfftn(a, s=None, axes=None, norm=None):
    # Check for types, no conversions for now
    fft_type = None
    if a.dtype == np.complex128:
        fft_type = FFTCode.FFT_Z2D
    elif a.dtype == np.complex64:
        fft_type = FFTCode.FFT_C2R
    else:
        raise TypeError("FFT input not supported, missing a conversion")

    s, axes = _sanitize_user_axes(a, s, axes)
    print('S {} AXES {} DIM {}'.format(s,axes, a.ndim))

    operate_by_axes = (len(axes) != len(set(axes))) or (len(axes) != a.ndim)
    if not operate_by_axes:
        operate_by_axes = axes != sorted(axes)
        
    # Operate by axes
    if operate_by_axes:
        if len(axes) > 1:
            c2r = a.fft(s=s[0:-1], axes=axes[0:-1], kind=_real_to_complex_kind(fft_type), direction=FFTDirection.INVERSE, norm=norm)
        else:
            c2r = a
        return c2r.fft(s=[s[-1]], axes=[axes[-1]], kind=fft_type, direction=FFTDirection.INVERSE, norm=norm)
    # Operate as a single FFT
    else:
        # cuFFT out-of-place C2R always overwrites the input buffer, which is not what we want here, so copy
        b = a.copy()
        return b.fft(s=s, axes=axes, kind=fft_type, direction=FFTDirection.INVERSE, norm=norm)


@add_boilerplate("a")
def hfft(a, n=None, axis=None, norm=None):
    s = (n,) if n is not None else None
    # Add checks to ensure input is hermitian?
    # Essentially a C2R FFT, with reverse sign (forward transform, forward norm)
    return irfftn(a=a.conjugate(), s=s, axes=axis, norm=FFTNormalization.reverse(norm))


@add_boilerplate("a")
def ihfft(a, n=None, axis=None, norm=None):
    s = (n,) if n is not None else None
    # Add checks to ensure input is hermitian?
    # Essentially a R2C FFT, with reverse sign (inverse transform, inverse norm)
    return rfftn(a=a, s=s, axes=axis, norm=FFTNormalization.reverse(norm)).conjugate()