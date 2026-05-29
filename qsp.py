from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

jax.config.update("jax_enable_x64", True)

# ---------------------------------------
# Returns F(Phi) for full or reduced Phi
# Implements Algorithm 3.2 from https://arxiv.org/pdf/2307.12468
#  - Phi: full phase factors of length d + 1 or reduced phase factors of length d_tilde
#  - parity: d mod 2 (default 1)
#  - full: True if Phi is full and False if Phi is reduced (default False)
#  - dtype: jnp.float32 or jnp.float64 (default jnp.float64)
# ---------------------------------------
@partial(jax.jit, static_argnames=("full", "parity", "dtype",))
def F(Phi, parity=1, full=False, dtype=jnp.float64):
    cdtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    if not full:
        # construct full phase factors using (2.5) in https://arxiv.org/pdf/2307.12468
        Phi = jnp.r_[Phi[::-1], Phi] if parity == 1 else jnp.r_[Phi[:0:-1], 2 * Phi[0], Phi[1:]]
    d = len(Phi) - 1
    I = jnp.array(1j, dtype=cdtype)

    n = 2 * d + 1
    j = jnp.arange(d + 1, dtype=dtype)
    x = (2.0 * jnp.pi / n) * j
    c = jnp.cos(x).astype(dtype)
    s = (I * jnp.sin(x)).astype(cdtype)
    
    w = jnp.exp(I * Phi).astype(cdtype)
    z = jnp.conj(w)

    a = jnp.full((d + 1,), w[0], dtype=cdtype)
    b = jnp.zeros((d + 1,), dtype=cdtype)

    def step(ab, wz):
        a, b = ab
        w, z = wz
        return ((a * c + b * s) * w, (a * s + b * c) * z), None

    (a, _), _ = lax.scan(step, (a, b), (w[1:], z[1:]))
    g = jnp.imag(a).astype(dtype)
    g = jnp.r_[g, g[d:0:-1]]
    v = jnp.fft.rfft(g, n).real.astype(dtype)

    f = v[0:d + 1:2] if parity == 0 else v[1:d + 1:2]
    if parity == 0:
        f = f.at[0].multiply(0.5)
    return (2.0 / n) * f

# ---------------------------------------
# Returns DF(Phi) for reduced Phi
# Implements Algorithm 3.3 from https://arxiv.org/pdf/2307.12468
#  - Phi: reduced phase factors of length d_tilde
#  - parity: d mod 2 (default 1)
#  - dtype: jnp.float32 or jnp.float64 (default jnp.float64)
# ---------------------------------------
@partial(jax.jit, static_argnames=("parity", "dtype",))
def DF(Phi, parity=1, dtype=jnp.float64):
    cdtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    I = jnp.array(1j, dtype=cdtype)
    
    d_tilde = Phi.size
    d = 2 * d_tilde - 2 + parity
    n = 2 * d + 1

    j = jnp.arange(d + 1, dtype=dtype)
    x = (2.0 * jnp.pi / n) * j
    c = jnp.cos(x).astype(dtype)
    s = (I * jnp.sin(x)).astype(cdtype)

    w = jnp.exp(I * Phi).astype(cdtype)
    z = jnp.conj(w)

    a = jnp.ones((d + 1,), dtype=cdtype)
    b = jnp.zeros((d + 1,), dtype=cdtype)

    def left(ab, wz):
        a, b = ab
        w, z = wz
        return (a * w * c + b * z * s, a * w * s + b * z * c), None

    (a, b), _ = lax.scan(left, (a, b), (w[1:][::-1], z[1:][::-1]))
    p = w[0] * a
    q = z[0] * b
    if parity == 1:
        p, q = c * p + s * q, s * p + c * q
    h = (2.0 * jnp.real(a * p - b * q)).astype(dtype)

    def body(carry, wz):
        a, b, p, q = carry
        w, z = wz
        a, b, p, q = (a * c - b * s) * z, (-a * s + b * c) * w, c * p * w + s * q * z, s * p * w + c * q * z
        return (a, b, p, q), (2.0 * jnp.real(a * p - b * q)).astype(dtype)

    (_, _, _, _), g = lax.scan(body, (a, b, p, q), (w[1:], z[1:]))
    g = jnp.concatenate([h[None, :], g], axis=0)
    g = jnp.concatenate([g, g[:, d:0:-1]], axis=1)
    v = jnp.fft.rfft(g, n, axis=1).real.astype(dtype)

    J = v[:, 0:d + 1:2] if parity == 0 else v[:, 1:d + 1:2]
    if parity == 0:
        J = J.at[:, 0].multiply(0.5)
    return (2.0 / n) * J.T

# ---------------------------------------
# Newton's method
# Implements Algorithm 3.1 from https://arxiv.org/pdf/2307.12468
#  - c: Chebyshev-coefficient vector of target polynomial
#  - T: number of iterations
#  - parity: d mod 2 (default 1)
# ---------------------------------------
@partial(jax.jit, static_argnames=("T", "parity"))
def newton(Phi_0, c, T, parity=1):
    def step(_, Phi):
        J = DF(Phi, parity)
        y = F(Phi, parity) - c
        return Phi - jnp.linalg.solve(J, y)
    return lax.fori_loop(0, T, step, Phi_0)

# ---------------------------------------
# Tests convergence of Algorithm 3.1 on random phase factors generated as follows:
# Sample uniformly from [-pi, pi]^d_tilde, then normalize to a certain ell_1 norm
#  - sigma: noise scale (default 0.1)
#  - T: number of Newton iterations (default 20)
#  - parity: d mod 2 (default 1)
#  - dtype: jnp.float32 or jnp.float64 (default jnp.float64) 
# ---------------------------------------
@partial(jax.jit, static_argnames=("d_tilde", "sigma", "T", "parity", "dtype"))
def test_ell_1(key, d_tilde, sigma=0.1, T=20, parity=1, dtype=jnp.float64):
    k1, k2 = jax.random.split(key, 2)

    Phi_star = jax.random.uniform(k1, (d_tilde,), minval=-jnp.pi, maxval=jnp.pi, dtype=dtype)
    # guaranteed contraction by Lemma 26 in https://arxiv.org/pdf/2209.10162
    Phi_star = 0.544 * Phi_star / jnp.linalg.norm(Phi_star, ord=1)
    c = F(Phi_star, parity)
    
    Phi_0 = Phi_star + (sigma / d_tilde) * jax.random.normal(k2, (d_tilde,), dtype=dtype)
    Phi_T = newton(Phi_0, c, T, parity)

    norm = jnp.linalg.norm(Phi_T, ord=1)
    ok = norm <= 1.0
    return ok, norm, Phi_0, Phi_T, Phi_star, c

# ---------------------------------------
# Tests convergence of Algorithm 3.1 on random phase factors sampled from B(0_d_tilde, r)
#  - r: radius of ball (default 1.0)
#  - sigma: noise scale (default 0.1)
#  - T: number of Newton iterations (default 20)
#  - parity: d mod 2 (default 1)
#  - dtype: jnp.float32 or jnp.float64 (default jnp.float64)
# ---------------------------------------
@partial(jax.jit, static_argnames=("d_tilde", "r", "sigma", "T", "parity", "dtype"))
def test_ell_2(key, d_tilde, r=1.0, sigma=0.1, T=20, parity=1, dtype=jnp.float64):
    k1, k2 = jax.random.split(key, 2)

    u = jax.random.normal(k1, (d_tilde + 2,), dtype=dtype)
    Phi_star = r * (u[:d_tilde] / jnp.linalg.norm(u))
    c = F(Phi_star, parity)

    Phi_0 = Phi_star + (sigma / d_tilde) * jax.random.normal(k2, (d_tilde,), dtype=dtype)
    Phi_T = newton(Phi_0, c, T, parity)

    norm = jnp.linalg.norm(Phi_T)
    ok = norm <= 1.0
    return ok, norm, Phi_0, Phi_T, Phi_star, c

if __name__ == "__main__":
    d_tilde = 100
    trials = 100_000

    # print passed every p trials
    p = 100

    # noise scale
    sigma = 0.1

    # ball radius
    r = 0.1

    # number of Newton iterations
    T = 20
    
    d_tildes = [4, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    for d in d_tildes:
        m = 0
        for t in range(trials):
            # samples Phi uniformly from B(r)
            key = jax.random.PRNGKey(t)
            k1, k2 = jax.random.split(key, 2)
            u = jax.random.normal(k1, (d + 2,))
            Phi = r * (u[:d] / jnp.linalg.norm(u))

            m = max(m, jnp.linalg.norm(F(Phi, 0) - 2 * Phi))
        print(d, m)