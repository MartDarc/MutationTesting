"""Lite version of scipy.linalg.

Notes
-----
This module is a lite version of the linalg.py module in SciPy which
contains high-level Python interface to the LAPACK library.  The lite
version only accesses the following LAPACK functions: dgesv, zgesv,
dgeev, zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf,
zgetrf, dpotrf, zpotrf, dgeqrf, zgeqrf, zungqr, dorgqr.
"""

__all__ = ['eigvalsh','svd', 'eig', 'eigh', 'norm','LinAlgError']

import functools
import operator
import warnings

from numpy.core import (
    array, asarray, zeros, empty, empty_like, intc, single, double,
    csingle, cdouble, inexact, complexfloating, newaxis, all, Inf, dot,
    add, multiply, sqrt, fastCopyAndTranspose, sum, isfinite,
    finfo, errstate, geterrobj, moveaxis, amin, amax, product, abs,
    atleast_2d, intp, asanyarray, object_, matmul,
    swapaxes, divide, count_nonzero, isnan, sign, argsort, sort
)
from numpy.core.multiarray import normalize_axis_index
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.lib.twodim_base import triu, eye
from numpy.linalg import lapack_lite, _umath_linalg


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.linalg')


fortran_int = intc


@set_module('numpy.linalg')
class LinAlgError(Exception):
    """
    Generic Python-exception-derived object raised by linalg functions.

    General purpose exception class, derived from Python's exception.Exception
    class, programmatically raised in linalg functions when a Linear
    Algebra-related condition would prevent further correct execution of the
    function.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> LA.inv(np.zeros((2,2)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "...linalg.py", line 350,
        in inv return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))
      File "...linalg.py", line 249,
        in solve
        raise LinAlgError('Singular matrix')
    numpy.linalg.LinAlgError: Singular matrix

    """


def _determine_error_states():
    errobj = geterrobj()
    bufsize = errobj[0]

    with errstate(invalid='call', over='ignore',
                  divide='ignore', under='ignore'):
        invalid_call_errmask = geterrobj()[1]

    return [bufsize, invalid_call_errmask, None]

# Dealing with errors in _umath_linalg
_linalg_error_extobj = _determine_error_states()
del _determine_error_states

def _raise_linalgerror_singular(err, flag):
    raise LinAlgError("Singular matrix")

def _raise_linalgerror_nonposdef(err, flag):
    raise LinAlgError("Matrix is not positive definite")

def _raise_linalgerror_eigenvalues_nonconvergence(err, flag):
    raise LinAlgError("Eigenvalues did not converge")

def _raise_linalgerror_svd_nonconvergence(err, flag):
    raise LinAlgError("SVD did not converge")

def _raise_linalgerror_lstsq(err, flag):
    raise LinAlgError("SVD did not converge in Linear Least Squares")

def _raise_linalgerror_qr(err, flag):
    raise LinAlgError("Incorrect argument found while performing "
                      "QR factorization")

def get_linalg_error_extobj(callback):
    extobj = list(_linalg_error_extobj)  # make a copy
    extobj[2] = callback
    return extobj

def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap

def isComplexType(t):
    return issubclass(t, complexfloating)

_real_types_map = {single : single,
                   double : double,
                   csingle : single,
                   cdouble : double}

_complex_types_map = {single : csingle,
                      double : cdouble,
                      csingle : csingle,
                      cdouble : cdouble}

def _realType(t, default=double):
    return _real_types_map.get(t, default)

def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)

def _commonType(*arrays):
    # in lite version, use higher precision (always double or cdouble)
    result_type = single
    is_complex = False
    for a in arrays:
        if issubclass(a.dtype.type, inexact):
            if isComplexType(a.dtype.type):
                is_complex = True
            rt = _realType(a.dtype.type, default=None)
            if rt is None:
                # unsupported inexact scalar
                raise TypeError("array type %s is unsupported in linalg" %
                        (a.dtype.name,))
        else:
            rt = double
        if rt is double:
            result_type = double
    if is_complex:
        t = cdouble
        result_type = _complex_types_map[result_type]
    else:
        t = double
    return t, result_type


# _fastCopyAndTranpose assumes the input is 2D (as all the calls in here are).

_fastCT = fastCopyAndTranspose

def _to_native_byte_order(*arrays):
    ret = []
    for arr in arrays:
        if arr.dtype.byteorder not in ('=', '|'):
            ret.append(asarray(arr, dtype=arr.dtype.newbyteorder('=')))
        else:
            ret.append(arr)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret

def _fastCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.type is not type:
            a = a.astype(type)
        cast_arrays = cast_arrays + (_fastCT(a),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays

def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'two-dimensional' % a.ndim)

def _assert_stacked_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'at least two-dimensional' % a.ndim)

def _assert_stacked_square(*arrays):
    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise LinAlgError('Last 2 dimensions of the array must be square')

def _assert_finite(*arrays):
    for a in arrays:
        if not isfinite(a).all():
            raise LinAlgError("Array must not contain infs or NaNs")

def _is_empty_2d(arr):
    # check size first for efficiency
    return arr.size == 0 and product(arr.shape[-2:]) == 0


def transpose(a):
    """
    Transpose each matrix in a stack of matrices.

    Unlike np.transpose, this only swaps the last two axes, rather than all of
    them

    Parameters
    ----------
    a : (...,M,N) array_like

    Returns
    -------
    aT : (...,N,M) ndarray
    """
    return swapaxes(a, -1, -2)

# Eigenvalues

def _unary_dispatcher(a):
    return (a,)

def _eigvalsh_dispatcher(a, UPLO=None):
    return (a,)

@array_function_dispatch(_eigvalsh_dispatcher)
def eigvalsh(a, UPLO='L'):
    """
    Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

    Main difference from eigh: the eigenvectors are not computed.

    Parameters
    ----------
    a : (..., M, M) array_like
        A complex- or real-valued matrix whose eigenvalues are to be
        computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will
        be considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    w : (..., M,) ndarray
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigh : eigenvalues and eigenvectors of real symmetric or complex Hermitian
           (conjugate symmetric) arrays.
    eigvals : eigenvalues of general real or complex arrays.
    eig : eigenvalues and right eigenvectors of general real or complex
          arrays.
    scipy.linalg.eigvalsh : Similar function in SciPy.

    Notes
    -----

    .. versionadded:: 1.8.0

    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The eigenvalues are computed using LAPACK routines ``_syevd``, ``_heevd``.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> LA.eigvalsh(a)
    array([ 0.17157288,  5.82842712]) # may vary

    >>> # demonstrate the treatment of the imaginary part of the diagonal
    >>> a = np.array([[5+2j, 9-2j], [0+2j, 2-1j]])
    >>> a
    array([[5.+2.j, 9.-2.j],
           [0.+2.j, 2.-1.j]])
    >>> # with UPLO='L' this is numerically equivalent to using LA.eigvals()
    >>> # with:
    >>> b = np.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])
    >>> b
    array([[5.+0.j, 0.-2.j],
           [0.+2.j, 2.+0.j]])
    >>> wa = LA.eigvalsh(a)
    >>> wb = LA.eigvals(b)
    >>> wa; wb
    array([1., 6.])
    array([6.+0.j, 1.+0.j])

    """
    UPLO = UPLO.upper()
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    extobj = get_linalg_error_extobj(
        _raise_linalgerror_eigenvalues_nonconvergence)
    if UPLO == 'L':
        gufunc = _umath_linalg.eigvalsh_lo
    else:
        gufunc = _umath_linalg.eigvalsh_up

    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->d' if isComplexType(t) else 'd->d'
    w = gufunc(a, signature=signature, extobj=extobj)
    return w.astype(_realType(result_t), copy=False)

def _convertarray(a):
    t, result_t = _commonType(a)
    a = _fastCT(a.astype(t))
    return a, t, result_t


# Eigenvectors


@array_function_dispatch(_unary_dispatcher)
def eig(a):
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    Parameters
    ----------
    a : (..., M, M) array
        Matrices for which the eigenvalues and right eigenvectors will
        be computed

    Returns
    -------
    w : (..., M) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be of complex type, unless the imaginary part is
        zero in which case it will be cast to a real type. When `a`
        is real the resulting eigenvalues will be real (0 imaginary
        part) or occur in conjugate pairs

    v : (..., M, M) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvals : eigenvalues of a non-symmetric array.
    eigh : eigenvalues and eigenvectors of a real symmetric or complex
           Hermitian (conjugate symmetric) array.
    eigvalsh : eigenvalues of a real symmetric or complex Hermitian
               (conjugate symmetric) array.
    scipy.linalg.eig : Similar function in SciPy that also solves the
                       generalized eigenvalue problem.
    scipy.linalg.schur : Best choice for unitary and other non-Hermitian
                         normal matrices.

    Notes
    -----

    .. versionadded:: 1.8.0

    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    This is implemented using the ``_geev`` LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.

    The number `w` is an eigenvalue of `a` if there exists a vector
    `v` such that ``a @ v = w * v``. Thus, the arrays `a`, `w`, and
    `v` satisfy the equations ``a @ v[:,i] = w[i] * v[:,i]``
    for :math:`i \\in \\{0,...,M-1\\}`.

    The array `v` of eigenvectors may not be of maximum rank, that is, some
    of the columns may be linearly dependent, although round-off error may
    obscure that fact. If the eigenvalues are all different, then theoretically
    the eigenvectors are linearly independent and `a` can be diagonalized by
    a similarity transformation using `v`, i.e, ``inv(v) @ a @ v`` is diagonal.

    For non-Hermitian normal matrices the SciPy function `scipy.linalg.schur`
    is preferred because the matrix `v` is guaranteed to be unitary, which is
    not the case when using `eig`. The Schur factorization produces an
    upper triangular matrix rather than a diagonal matrix, but for normal
    matrices only the diagonal of the upper triangular matrix is needed, the
    rest is roundoff error.

    Finally, it is emphasized that `v` consists of the *right* (as in
    right-hand side) eigenvectors of `a`.  A vector `y` satisfying
    ``y.T @ a = z * y.T`` for some number `z` is called a *left*
    eigenvector of `a`, and, in general, the left and right eigenvectors
    of a matrix are not necessarily the (perhaps conjugate) transposes
    of each other.

    References
    ----------
    G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando, FL,
    Academic Press, Inc., 1980, Various pp.

    Examples
    --------
    >>> from numpy import linalg as LA

    (Almost) trivial example with real e-values and e-vectors.

    >>> w, v = LA.eig(np.diag((1, 2, 3)))
    >>> w; v
    array([1., 2., 3.])
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Real matrix possessing complex e-values and e-vectors; note that the
    e-values are complex conjugates of each other.

    >>> w, v = LA.eig(np.array([[1, -1], [1, 1]]))
    >>> w; v
    array([1.+1.j, 1.-1.j])
    array([[0.70710678+0.j        , 0.70710678-0.j        ],
           [0.        -0.70710678j, 0.        +0.70710678j]])

    Complex-valued matrix with real e-values (but complex-valued e-vectors);
    note that ``a.conj().T == a``, i.e., `a` is Hermitian.

    >>> a = np.array([[1, 1j], [-1j, 1]])
    >>> w, v = LA.eig(a)
    >>> w; v
    array([2.+0.j, 0.+0.j])
    array([[ 0.        +0.70710678j,  0.70710678+0.j        ], # may vary
           [ 0.70710678+0.j        , -0.        +0.70710678j]])

    Be careful about round-off error!

    >>> a = np.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])
    >>> # Theor. e-values are 1 +/- 1e-9
    >>> w, v = LA.eig(a)
    >>> w; v
    array([1., 1.])
    array([[1., 0.],
           [0., 1.]])

    """
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    _assert_finite(a)
    t, result_t = _commonType(a)

    extobj = get_linalg_error_extobj(
        _raise_linalgerror_eigenvalues_nonconvergence)
    signature = 'D->DD' if isComplexType(t) else 'd->DD'
    w, vt = _umath_linalg.eig(a, signature=signature, extobj=extobj)

    if not isComplexType(t) and all(w.imag == 0.0):
        w = w.real
        vt = vt.real
        result_t = _realType(result_t)
    else:
        result_t = _complexType(result_t)

    vt = vt.astype(result_t, copy=False)
    return w.astype(result_t, copy=False), wrap(vt)


@array_function_dispatch(_eigvalsh_dispatcher)
def eigh(a, UPLO='L'):
    """
    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    Parameters
    ----------
    a : (..., M, M) array
        Hermitian or real symmetric matrices whose eigenvalues and
        eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will
        be considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    w : (..., M) ndarray
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.
    v : {(..., M, M) ndarray, (..., M, M) matrix}
        The column ``v[:, i]`` is the normalized eigenvector corresponding
        to the eigenvalue ``w[i]``.  Will return a matrix object if `a` is
        a matrix object.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvalsh : eigenvalues of real symmetric or complex Hermitian
               (conjugate symmetric) arrays.
    eig : eigenvalues and right eigenvectors for non-symmetric arrays.
    eigvals : eigenvalues of non-symmetric arrays.
    scipy.linalg.eigh : Similar function in SciPy (but also solves the
                        generalized eigenvalue problem).

    Notes
    -----

    .. versionadded:: 1.8.0

    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The eigenvalues/eigenvectors are computed using LAPACK routines ``_syevd``,
    ``_heevd``.

    The eigenvalues of real symmetric or complex Hermitian matrices are
    always real. [1]_ The array `v` of (column) eigenvectors is unitary
    and `a`, `w`, and `v` satisfy the equations
    ``dot(a, v[:, i]) = w[i] * v[:, i]``.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pg. 222.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> a
    array([[ 1.+0.j, -0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> w, v = LA.eigh(a)
    >>> w; v
    array([0.17157288, 5.82842712])
    array([[-0.92387953+0.j        , -0.38268343+0.j        ], # may vary
           [ 0.        +0.38268343j,  0.        -0.92387953j]])

    >>> np.dot(a, v[:, 0]) - w[0] * v[:, 0] # verify 1st e-val/vec pair
    array([5.55111512e-17+0.0000000e+00j, 0.00000000e+00+1.2490009e-16j])
    >>> np.dot(a, v[:, 1]) - w[1] * v[:, 1] # verify 2nd e-val/vec pair
    array([0.+0.j, 0.+0.j])

    >>> A = np.matrix(a) # what happens if input is a matrix object
    >>> A
    matrix([[ 1.+0.j, -0.-2.j],
            [ 0.+2.j,  5.+0.j]])
    >>> w, v = LA.eigh(A)
    >>> w; v
    array([0.17157288, 5.82842712])
    matrix([[-0.92387953+0.j        , -0.38268343+0.j        ], # may vary
            [ 0.        +0.38268343j,  0.        -0.92387953j]])

    >>> # demonstrate the treatment of the imaginary part of the diagonal
    >>> a = np.array([[5+2j, 9-2j], [0+2j, 2-1j]])
    >>> a
    array([[5.+2.j, 9.-2.j],
           [0.+2.j, 2.-1.j]])
    >>> # with UPLO='L' this is numerically equivalent to using LA.eig() with:
    >>> b = np.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])
    >>> b
    array([[5.+0.j, 0.-2.j],
           [0.+2.j, 2.+0.j]])
    >>> wa, va = LA.eigh(a)
    >>> wb, vb = LA.eig(b)
    >>> wa; wb
    array([1., 6.])
    array([6.+0.j, 1.+0.j])
    >>> va; vb
    array([[-0.4472136 +0.j        , -0.89442719+0.j        ], # may vary
           [ 0.        +0.89442719j,  0.        -0.4472136j ]])
    array([[ 0.89442719+0.j       , -0.        +0.4472136j],
           [-0.        +0.4472136j,  0.89442719+0.j       ]])
    """
    UPLO = UPLO.upper()
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)

    extobj = get_linalg_error_extobj(
        _raise_linalgerror_eigenvalues_nonconvergence)
    if UPLO == 'L':
        gufunc = _umath_linalg.eigh_lo
    else:
        gufunc = _umath_linalg.eigh_up

    signature = 'D->dD' if isComplexType(t) else 'd->dd'
    w, vt = gufunc(a, signature=signature, extobj=extobj)
    w = w.astype(_realType(result_t), copy=False)
    vt = vt.astype(result_t, copy=False)
    return w, wrap(vt)


# Singular value decomposition

def _svd_dispatcher(a, full_matrices=None, compute_uv=None, hermitian=None):
    return (a,)


@array_function_dispatch(_svd_dispatcher)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """
    Singular Value Decomposition.

    When `a` is a 2D array, it is factorized as ``u @ np.diag(s) @ vh
    = (u * s) @ vh``, where `u` and `vh` are 2D unitary arrays and `s` is a 1D
    array of `a`'s singular values. When `a` is higher-dimensional, SVD is
    applied in stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) array_like
        A real or complex array with ``a.ndim >= 2``.
    full_matrices : bool, optional
        If True (default), `u` and `vh` have the shapes ``(..., M, M)`` and
        ``(..., N, N)``, respectively.  Otherwise, the shapes are
        ``(..., M, K)`` and ``(..., K, N)``, respectively, where
        ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether or not to compute `u` and `vh` in addition to `s`.  True
        by default.
    hermitian : bool, optional
        If True, `a` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.

        .. versionadded:: 1.17.0

    Returns
    -------
    u : { (..., M, M), (..., M, K) } array
        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`. The size of the last two dimensions
        depends on the value of `full_matrices`. Only returned when
        `compute_uv` is True.
    s : (..., K) array
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    vh : { (..., N, N), (..., K, N) } array
        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`. The size of the last two dimensions
        depends on the value of `full_matrices`. Only returned when
        `compute_uv` is True.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See Also
    --------
    scipy.linalg.svd : Similar function in SciPy.
    scipy.linalg.svdvals : Compute singular values of a matrix.

    Notes
    -----

    .. versionchanged:: 1.8.0
       Broadcasting rules apply, see the `numpy.linalg` documentation for
       details.

    The decomposition is performed using LAPACK routine ``_gesdd``.

    SVD is usually described for the factorization of a 2D matrix :math:`A`.
    The higher-dimensional case will be discussed below. In the 2D case, SVD is
    written as :math:`A = U S V^H`, where :math:`A = a`, :math:`U= u`,
    :math:`S= \\mathtt{np.diag}(s)` and :math:`V^H = vh`. The 1D array `s`
    contains the singular values of `a` and `u` and `vh` are unitary. The rows
    of `vh` are the eigenvectors of :math:`A^H A` and the columns of `u` are
    the eigenvectors of :math:`A A^H`. In both cases the corresponding
    (possibly non-zero) eigenvalues are given by ``s**2``.

    If `a` has more than two dimensions, then broadcasting rules apply, as
    explained in :ref:`routines.linalg-broadcasting`. This means that SVD is
    working in "stacked" mode: it iterates over all indices of the first
    ``a.ndim - 2`` dimensions and for each combination SVD is applied to the
    last two indices. The matrix `a` can be reconstructed from the
    decomposition with either ``(u * s[..., None, :]) @ vh`` or
    ``u @ (s[..., None] * vh)``. (The ``@`` operator can be replaced by the
    function ``np.matmul`` for python versions below 3.5.)

    If `a` is a ``matrix`` object (as opposed to an ``ndarray``), then so are
    all the return values.

    Examples
    --------
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    >>> b = np.random.randn(2, 7, 8, 3) + 1j*np.random.randn(2, 7, 8, 3)

    Reconstruction based on full SVD, 2D case:

    >>> u, s, vh = np.linalg.svd(a, full_matrices=True)
    >>> u.shape, s.shape, vh.shape
    ((9, 9), (6,), (6, 6))
    >>> np.allclose(a, np.dot(u[:, :6] * s, vh))
    True
    >>> smat = np.zeros((9, 6), dtype=complex)
    >>> smat[:6, :6] = np.diag(s)
    >>> np.allclose(a, np.dot(u, np.dot(smat, vh)))
    True

    Reconstruction based on reduced SVD, 2D case:

    >>> u, s, vh = np.linalg.svd(a, full_matrices=False)
    >>> u.shape, s.shape, vh.shape
    ((9, 6), (6,), (6, 6))
    >>> np.allclose(a, np.dot(u * s, vh))
    True
    >>> smat = np.diag(s)
    >>> np.allclose(a, np.dot(u, np.dot(smat, vh)))
    True

    Reconstruction based on full SVD, 4D case:

    >>> u, s, vh = np.linalg.svd(b, full_matrices=True)
    >>> u.shape, s.shape, vh.shape
    ((2, 7, 8, 8), (2, 7, 3), (2, 7, 3, 3))
    >>> np.allclose(b, np.matmul(u[..., :3] * s[..., None, :], vh))
    True
    >>> np.allclose(b, np.matmul(u[..., :3], s[..., None] * vh))
    True

    Reconstruction based on reduced SVD, 4D case:

    >>> u, s, vh = np.linalg.svd(b, full_matrices=False)
    >>> u.shape, s.shape, vh.shape
    ((2, 7, 8, 3), (2, 7, 3), (2, 7, 3, 3))
    >>> np.allclose(b, np.matmul(u * s[..., None, :], vh))
    True
    >>> np.allclose(b, np.matmul(u, s[..., None] * vh))
    True

    """
    import numpy as _nx
    a, wrap = _makearray(a)

    if hermitian:
        # note: lapack svd returns eigenvalues with s ** 2 sorted descending,
        # but eig returns s sorted ascending, so we re-order the eigenvalues
        # and related arrays to have the correct order
        if compute_uv:
            s, u = eigh(a)
            sgn = sign(s)
            s = abs(s)
            sidx = argsort(s)[..., ::-1]
            sgn = _nx.take_along_axis(sgn, sidx, axis=-1)
            s = _nx.take_along_axis(s, sidx, axis=-1)
            u = _nx.take_along_axis(u, sidx[..., None, :], axis=-1)
            # singular values are unsigned, move the sign into v
            vt = transpose(u * sgn[..., None, :]).conjugate()
            return wrap(u), s, wrap(vt)
        else:
            s = eigvalsh(a)
            s = s[..., ::-1]
            s = abs(s)
            return sort(s)[..., ::-1]

    _assert_stacked_2d(a)
    t, result_t = _commonType(a)

    extobj = get_linalg_error_extobj(_raise_linalgerror_svd_nonconvergence)

    m, n = a.shape[-2:]
    if compute_uv:
        if full_matrices:
            if m < n:
                gufunc = _umath_linalg.svd_m_f
            else:
                gufunc = _umath_linalg.svd_n_f
        else:
            if m < n:
                gufunc = _umath_linalg.svd_m_s
            else:
                gufunc = _umath_linalg.svd_n_s

        signature = 'D->DdD' if isComplexType(t) else 'd->ddd'
        u, s, vh = gufunc(a, signature=signature, extobj=extobj)
        u = u.astype(result_t, copy=False)
        s = s.astype(_realType(result_t), copy=False)
        vh = vh.astype(result_t, copy=False)
        return wrap(u), s, wrap(vh)
    else:
        if m < n:
            gufunc = _umath_linalg.svd_m
        else:
            gufunc = _umath_linalg.svd_n

        signature = 'D->d' if isComplexType(t) else 'd->d'
        s = gufunc(a, signature=signature, extobj=extobj)
        s = s.astype(_realType(result_t), copy=False)
        return s


def _multi_svd_norm(x, row_axis, col_axis, op):
    """Compute a function of the singular values of the 2-D matrices in `x`.

    This is a private utility function used by `numpy.linalg.norm()`.

    Parameters
    ----------
    x : ndarray
    row_axis, col_axis : int
        The axes of `x` that hold the 2-D matrices.
    op : callable
        This should be either numpy.amin or `numpy.amax` or `numpy.sum`.

    Returns
    -------
    result : float or ndarray
        If `x` is 2-D, the return values is a float.
        Otherwise, it is an array with ``x.ndim - 2`` dimensions.
        The return values are either the minimum or maximum or sum of the
        singular values of the matrices, depending on whether `op`
        is `numpy.amin` or `numpy.amax` or `numpy.sum`.

    """
    y = moveaxis(x, (row_axis, col_axis), (-2, -1))
    result = op(svd(y, compute_uv=False), axis=-1)
    return result


def _norm_dispatcher(x, ord=None, axis=None, keepdims=None):
    return (x,)


@array_function_dispatch(_norm_dispatcher)
def norm(x, ord=None, axis=None, keepdims=False):
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``x.ravel`` will be returned.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.
    axis : {None, int, 2-tuple of ints}, optional.
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
        is None.

        .. versionadded:: 1.8.0

    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

        .. versionadded:: 1.10.0

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    See Also
    --------
    scipy.linalg.norm : Similar function in SciPy.

    Notes
    -----
    For values of ``ord < 1``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices and raise a ValueError when ``x.ndim != 2``.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, ...,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> LA.norm(a)
    7.745966692414834
    >>> LA.norm(b)
    7.745966692414834
    >>> LA.norm(b, 'fro')
    7.745966692414834
    >>> LA.norm(a, np.inf)
    4.0
    >>> LA.norm(b, np.inf)
    9.0
    >>> LA.norm(a, -np.inf)
    0.0
    >>> LA.norm(b, -np.inf)
    2.0

    >>> LA.norm(a, 1)
    20.0
    >>> LA.norm(b, 1)
    7.0
    >>> LA.norm(a, -1)
    -4.6566128774142013e-010
    >>> LA.norm(b, -1)
    6.0
    >>> LA.norm(a, 2)
    7.745966692414834
    >>> LA.norm(b, 2)
    7.3484692283495345

    >>> LA.norm(a, -2)
    0.0
    >>> LA.norm(b, -2)
    1.8570331885190563e-016 # may vary
    >>> LA.norm(a, 3)
    5.8480354764257312 # may vary
    >>> LA.norm(a, -3)
    0.0

    Using the `axis` argument to compute vector norms:

    >>> c = np.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> LA.norm(c, axis=0)
    array([ 1.41421356,  2.23606798,  5.        ])
    >>> LA.norm(c, axis=1)
    array([ 3.74165739,  4.24264069])
    >>> LA.norm(c, ord=1, axis=1)
    array([ 6.,  6.])

    Using the `axis` argument to compute matrix norms:

    >>> m = np.arange(8).reshape(2,2,2)
    >>> LA.norm(m, axis=(1,2))
    array([  3.74165739,  11.22497216])
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (3.7416573867739413, 11.224972160321824)

    """
    x = asarray(x)

    if not issubclass(x.dtype.type, (inexact, object_)):
        x = x.astype(float)

    # Immediately handle some default, simple, fast, and common cases.
    if axis is None:
        ndim = x.ndim
        if ((ord is None) or
            (ord in ('f', 'fro') and ndim == 2) or
            (ord == 2 and ndim == 1)):

            x = x.ravel(order='K')
            if isComplexType(x.dtype.type):
                sqnorm = dot(x.real, x.real) + dot(x.imag, x.imag)
            else:
                sqnorm = dot(x, x)
            ret = sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim*[1])
            return ret

    # Normalize the `axis` argument to a tuple.
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception as e:
            raise TypeError("'axis' must be None, an integer or a tuple of integers") from e
        axis = (axis,)

    if len(axis) == 1:
        if ord == Inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -Inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Zero norm
            return (x != 0).astype(x.real.dtype).sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            # special case for speedup
            return add.reduce(abs(x), axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # special case for speedup
            s = (x.conj() * x).real
            return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
        # None of the str-type keywords for ord ('fro', 'nuc')
        # are valid for vectors
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = abs(x)
            absx **= ord
            ret = add.reduce(absx, axis=axis, keepdims=keepdims)
            ret **= (1 / ord)
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        row_axis = normalize_axis_index(row_axis, nd)
        col_axis = normalize_axis_index(col_axis, nd)
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            ret =  _multi_svd_norm(x, row_axis, col_axis, amax)
        elif ord == -2:
            ret = _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).max(axis=col_axis)
        elif ord == Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
        elif ord == -Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            ret = sqrt(add.reduce((x.conj() * x).real, axis=axis))
        elif ord == 'nuc':
            ret = _multi_svd_norm(x, row_axis, col_axis, sum)
        else:
            raise ValueError("Invalid norm order for matrices.")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")

