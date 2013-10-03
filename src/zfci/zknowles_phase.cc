//
// BAGEL - Parallel electron correlation program.
// Filename: zknowles_phase.cc
// Copyright (C) 2013 Michael Caldwell
//
// Author: Michael Caldwell <caldwell@u.northwestern.edu>>
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

#include <cmath>
#include <random>
#include <src/math/algo.h>
#include <src/zfci/zknowles.h>

using namespace std;
using namespace bagel;

void ZKnowlesHandy::mult_phase_factor() {
  cout << "         "  << "Applying random phase factor to integrals" << endl << endl;
  const size_t norb2 = norb_*norb_;
  const size_t norb3 = norb2*norb_;
  const size_t norb4 = norb3*norb_;

  auto phase = make_shared<ZMatrix>(norb_, norb_);
  for (int i = 0; i != norb_; ++i) {
    const double ran = rand();
    const complex<double> fac(cos(ran), sin(ran));
    phase->element(i,i) = fac;
  }

  //transform 1e integrals.
  auto mo1e = make_shared<ZMatrix>(norb_, norb_);
  copy_n(jop_->mo1e_ptr(), norb2, mo1e->data());
  *mo1e = *phase % *mo1e * *phase;

  //transforming 4 index mo2e
  auto mo2e = make_shared<ZMatrix>(norb2, norb2, true);
  unique_ptr<complex<double>[]> tmp(new complex<double>[norb4]);
  unique_ptr<complex<double>[]> trans(new complex<double>[norb4]);

  // 1) make (i*, j, k, l) (left multiply by phase*)
  zgemm3m_("c","n", norb_, norb3, norb_, 1.0, phase->data(), norb_, jop_->mo2e_ptr(), norb_, 0.0, trans.get(), norb_);

  // 2) make (i*, j, k, l') (right multiply by phase)
  zgemm3m_("n","n", norb3, norb_, norb_, 1.0, trans.get(), norb3, phase->data(), norb_, 0.0, tmp.get(), norb3);

  // 3) transpose to make (k, l', i*, j) now (kl|ij)
  mytranspose_(tmp.get(), norb2, norb2, trans.get());

  // 4) make (k*, l', i*, j) (left multiply by phase*)
  zgemm3m_("c","n", norb_, norb3, norb_, 1.0, phase->data(), norb_, trans.get(), norb_, 0.0, tmp.get(), norb_);

  // 5) make (k*, l', i*, j') (right multiply by phase)
  zgemm3m_("n","n", norb3, norb_, norb_, 1.0, tmp.get(), norb3, phase->data(), norb_, 0.0, mo2e->data(), norb3);

  // set mo1e and mo2e
  jop_->set_moints(mo1e, mo2e);
}
