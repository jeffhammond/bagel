//
// BAGEL - Parallel electron correlation program.
// Filename: jvec.cc
// Copyright (C) 2012 Toru Shiozaki
//
// Author: Toru Shiozaki <shiozaki@northwestern.edu>
// Maintainer: Shiozaki group
//
// This file is part of the BAGEL package.
//
// The BAGEL package is free software; you can redistribute it and/or modify
// it under the terms of the GNU Library General Public License as published by
// the Free Software Foundation; either version 3, or (at your option)
// any later version.
//
// The BAGEL package is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Library General Public License for more details.
//
// You should have received a copy of the GNU Library General Public License
// along with the BAGEL package; see COPYING.  If not, write to
// the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
//


#include <src/casscf/jvec.h>

using namespace std;
using namespace bagel;

Jvec::Jvec(shared_ptr<FCI> fci, shared_ptr<const Coeff> coeff, const size_t nclosed, const size_t nact, const size_t nvirt) {

  shared_ptr<const DFDist> df = fci->geom()->df();
  const size_t nocc = nclosed+nact;
  shared_ptr<const Matrix> cc = coeff->slice(0, nocc);

  shared_ptr<RDM<1>> rdm1_av = fci->rdm1_av();
  shared_ptr<RDM<2>> rdm = fci->rdm2_av();

  // half_ = J^{-1/2}(D|ix)
  half_ = df->compute_half_transform(cc)->apply_J();

  // jvec_ = J^{-1/2} (D|ij) Gamma_ij,kl
  {
    shared_ptr<DFFullDist> in = half_->compute_second_transform(cc);

    // TODO this is not a very efficient implementation, obviously
    // for the time being, I form the entire 2RDM
    rdm2all_ = std::make_shared<btas::Tensor4<double>>(btas::Range(btas::Range1(nocc),4));
    fill(rdm2all_->begin(), rdm2all_->end(), 0.0);
    {
      // closed-closed
      for (int i = 0; i != nclosed; ++i) {
        for (int j = 0; j != nclosed; ++j) {
          (*rdm2all_)(j,j,i,i) += 4.0;
          (*rdm2all_)(j,i,i,j) -= 2.0;
        }
      }
      // active-active
      for (int i = 0; i != nact; ++i) {
        for (int j = 0; j != nact; ++j) {
          for (int k = 0; k != nact; ++k) {
            copy_n(rdm->data()+nact*(k+nact*(j+nact*i)), nact, rdm2all_->data()+nclosed+nocc*(k+nclosed+nocc*(j+nclosed+nocc*(i+nclosed))));
          }
        }
      }
      // active-closed
      for (int i = 0; i != nclosed; ++i) {
        for (int j = 0; j != nact; ++j) {
          for (int k = 0; k != nact; ++k) {
            (*rdm2all_)(i, i, k+nclosed, j+nclosed) += rdm1_av->element(k,j) * 2.0;
            (*rdm2all_)(k+nclosed, j+nclosed, i, i) += rdm1_av->element(k,j) * 2.0;
            (*rdm2all_)(i, k+nclosed, j+nclosed, i) -= rdm1_av->element(k,j);
            (*rdm2all_)(k+nclosed, i, i, j+nclosed) -= rdm1_av->element(k,j);
          }
        }
      }
    }
    jvec_ = in->apply_2rdm(rdm2all_)->apply_J();
#if 0
double* d = rdm2_all_.get();
for (int i = 0; i != nocc; ++i) {
for (int j = 0; j != nocc; ++j) {
for (int k = 0; k != nocc; ++k) {
for (int l = 0; l != nocc; ++l, ++d) {
if (abs(*d) > 1.0e-10) cout << i << " " << j << " " << k << " " << l << " " << setprecision(10) << *d << endl;
}}}}
#endif
  }

};
