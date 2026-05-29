# QSVT Algorithms and Experiments

This repository contains JAX implementations of Algorithms 3.1, 3.2, and 3.3 from [abs/2307.12468](https://arxiv.org/abs/2307.12468), with additional randomized numerical experiments.

## Features

* Fast evaluation of the phase factors to Chebyshev coefficients map `F(Phi)` for both full and reduced phase factors
* Fast Jacobian computation `DF(Phi)` for reduced phase factors
* Newton's method for finding reduce phase factors that induce target Chebyshev coefficients
* Randomized convergence tests for phase factors sampled from `ell_1` and `ell_2` balls
* JAX JIT compilation and optional 64-bit precision

## Requirements

```bash
pip install jax jaxlib
```

The code enables 64-bit mode by default:

```python
jax.config.update("jax_enable_x64", True)
```

## Main Functions

### `F(Phi, parity=1, full=False, dtype=jnp.float64)`

Evaluates the Chebyshev coefficients associated with the provided phase factors, either full or reduced.

```python
c = F(Phi, parity=1)
```

Arguments:

* `Phi`: phase factors

  * if `full=False`, `Phi` is interpreted as reduced phase factors
  * if `full=True`, `Phi` is interpreted as full phase factors
* `parity`: parity of the target degree `d`, equal to `d mod 2`, default `1`
* `full`: whether `Phi` is full, default `False`
* `dtype`: floating-point precision, default `jnp.float64`

Returns:

* Chebyshev coefficients corresponding to the phase factors.

### `DF(Phi, parity=1, dtype=jnp.float64)`

Computes the Jacobian of `F` at the provided reduced phase factors.

```python
J = DF(Phi, parity=1)
```

Arguments:

* `Phi`: reduced phase factors
* `parity`: parity of the target degree `d`, equal to `d mod 2`, default `1`
* `dtype`: floating-point precision, default `jnp.float64`

Returns:

* Jacobian matrix `J` whose columns correspond to derivatives with respect to the reduced phase factors.

### `newton(Phi_0, c, T, parity=1)`

Runs Newton's method to solve

```python
F(Phi) = c
```

starting from an initial guess `Phi_0`.

```python
Phi_T = newton(Phi_0, c, T=20, parity=1)
```

Arguments:

* `Phi_0`: initial guess, reduced phase factors
* `c`: target Chebyshev coefficients
* `T`: number of Newton iterations
* `parity`: parity of the target degree `d`, equal to `d mod 2`, default `1`

Returns:

* The phase factors after `T` iterations.

## Numerical Experiments

### `test_ell_1`

Tests the convergence of Newton's method on phase factors sampled from `[-pi, pi]^d_tilde` and then normalized to a target `ell_1` norm.

```python
ok, norm, Phi_0, Phi_T, Phi_star, c = test_ell_1(
    key,
    d_tilde=100,
    sigma=0.1,
    T=20,
    parity=1,
)
```

The test:

1. samples `Phi_star` uniformly from `[-pi, pi]^d_tilde` and then normalizes to have `ell_1` norm 0.544
2. computes `c = F(Phi_star)`
3. adds isotropic Gaussian noise of scale `sigma` to `Phi_star` to form the initial guess `Phi_0`
4. runs Newton's method for `T` iterations
5. checks convergence by ensuring `Phi_T` has `ell_1` norm at most `1`, since this blows up otherwise

### `test_ell_2`

Tests the convergence of Newton's method on phase factors sampled from a Euclidean ball of radius `r`.

```python
ok, norm, Phi_0, Phi_T, Phi_star, c = test_ell_2(
    key,
    d_tilde=100,
    r=1.0,
    sigma=0.1,
    T=20,
    parity=1,
)
```

The test:

1. samples `Phi_star` uniformly from the zero-centered Euclidean ball of radius `r`
2. computes `c = F(Phi_star)`
3. adds isotropic Gaussian noise of scale `sigma` to `Phi_star` to form the initial guess `Phi_0`
4. runs Newton's method for `T` iterations
5. checks convergence by ensuring `Phi_T` has `ell_2` norm at most `1`, since this blows up otherwise

## Script Mode

Running the file directly executes a scaling experiment over several reduced dimensions.

```bash
python qsp.py
```

The default experiment uses

```python
trials = 100_000
r = 0.1
d_tildes = [4, 20, 50, 100, 200, 500, 1000, 2000, 5000]
```

For each reduced dimension in `d_tildes`, the script samples random phase factors from a Euclidean ball of radius `r = 0.1` and prints the maximum observed deviation

```python
||F(Phi) - 2 Phi||_2
```

over all `trials = 100_000` trials.

## Notes on Precision

The implementation defaults to `jnp.float64` and complex128 arithmetic. This is recommended for experiments, since the Jacobian solve may become numerically sensitive.

To use single precision, pass

```python
dtype=jnp.float32
```

to `F`, `DF`, `test_ell_1`, or `test_ell_2`.

## References

The main algorithms implemented here are based on Algorithms 3.1, 3.2, and 3.3 from [abs/2307.12468](https://arxiv.org/abs/2307.12468)

The commented guaranteed contraction regime in `test_ell_1` references Lemma 26 from [abs/2209.10162](https://arxiv.org/abs/2209.10162)
