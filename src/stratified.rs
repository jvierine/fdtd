//! Shared stratified Yee field steps. Cold and coupled runs call these routines.
pub type Complex = [f64; 2];
#[inline(always)]
fn add(a: Complex, b: Complex) -> Complex {
    [a[0] + b[0], a[1] + b[1]]
}
#[inline(always)]
fn sub(a: Complex, b: Complex) -> Complex {
    [a[0] - b[0], a[1] - b[1]]
}
#[inline(always)]
fn mul(a: Complex, b: f64) -> Complex {
    [a[0] * b, a[1] * b]
}
#[inline(always)]
fn ik(a: Complex, k: f64) -> Complex {
    [-k * a[1], k * a[0]]
}
#[inline(always)]
pub fn phase(a: Complex, p: f64) -> Complex {
    let (s, c) = p.sin_cos();
    [a[0] * c - a[1] * s, a[0] * s + a[1] * c]
}
pub fn advance_magnetic(
    e: &[Vec<Complex>],
    h: &mut [Vec<Complex>],
    widths: &[f64],
    kx: f64,
    dt: f64,
    sponge: &[Vec<f64>],
) {
    let n = widths.len();
    for i in 0..n {
        let ip = (i + 1) % n;
        let df = |q: usize| mul(sub(e[q][ip], e[q][i]), 1. / widths[i]);
        let curl = [
            mul(df(1), -1.),
            sub(df(0), ik(e[2][i], kx)),
            ik(e[1][i], kx),
        ];
        for q in 0..3 {
            let d = 0.5 * dt * sponge[q + 3][i];
            h[q][i] = mul(sub(mul(h[q][i], 1. - d), mul(curl[q], dt)), 1. / (1. + d));
        }
    }
}
pub fn advance_electric(
    e: &mut [Vec<Complex>],
    h: &[Vec<Complex>],
    current: &[Vec<Complex>],
    wp: &[Vec<f64>],
    widths: &[f64],
    kx: f64,
    dt: f64,
    sponge: &[Vec<f64>],
    source: &[Vec<Complex>],
    source_phase: f64,
) {
    let n = widths.len();
    for i in 0..n {
        let im = (i + n - 1) % n;
        let db = |q: usize| mul(sub(h[q][i], h[q][im]), 2. / (widths[i] + widths[im]));
        let curl = [
            mul(db(1), -1.),
            sub(db(0), ik(h[2][i], kx)),
            ik(h[1][i], kx),
        ];
        for q in 0..3 {
            let d = 0.5 * dt * sponge[q][i];
            let drive = add(
                sub(curl[q], mul(current[q][i], wp[q][i])),
                phase(source[q][i], source_phase),
            );
            e[q][i] = mul(add(mul(e[q][i], 1. - d), mul(drive, dt)), 1. / (1. + d));
        }
    }
}
