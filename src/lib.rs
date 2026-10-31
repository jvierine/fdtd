//! Trapezoidal electron-current update shared by the 3-D and stratified solvers.
//! Internal time is c*t and current is scaled by the plasma frequency.
pub mod stratified;
#[inline(always)]
pub fn current_rhs(old: f64, electric_drive: f64, gyro_old: f64, dt: f64, damping: f64) -> f64 {
    (1.0-damping)*old + dt*electric_drive + 0.5*dt*gyro_old
}
#[inline(always)]
pub fn current_next(rhs: f64, gyro_new: f64, dt: f64, damping: f64) -> f64 {
    (rhs+0.5*dt*gyro_new)/(1.0+damping)
}
