#include <src/rysint/_vrr_drv.h>
void bagel::VRR_Driver::vrr_driver_6_2_1_1(double* out, const double* const roots, const double* const weights, const double& coeff,
    const std::array<double,3>& a, const std::array<double,3>& b, const std::array<double,3>& c, const std::array<double,3>& d,
    const double* const p, const double* const q, const double& xp, const double& xq, 
    const int* const amap, const int* const cmap, const int& asize_, double* const workx, double* const worky, double* const workz) {
  bagel::vrr_driver<6,2,1,1,6>(out, roots, weights, coeff, a, b, c, d, p, q, xp, xq, amap, cmap, asize_, workx, worky, workz);
}
