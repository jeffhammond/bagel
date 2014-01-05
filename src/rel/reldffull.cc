//
// BAGEL - Parallel electron correlation program.
// Filename: reldffull.cc
// Copyright (C) 2013 Toru Shiozaki
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


#include <src/rel/reldffull.h>

using namespace std;
using namespace bagel;

RelDFFull::RelDFFull(shared_ptr<const RelDFHalf> df, array<shared_ptr<const Matrix>,4> rcoeff, array<shared_ptr<const Matrix>,4> icoeff) : RelDFBase(*df) {

  basis_ = df->basis();
  if (basis_.size() != 1)
    throw logic_error("RelDFFull should be called with basis_.size() == 1");

  const int index = basis_.front()->basis(1);

  // TODO this could be cheaper by using a zgemm3m-type algorithm
  dffull_[0] = df->get_real()->compute_second_transform(rcoeff[index]);
  dffull_[0]->ax_plus_y(-1.0, df->get_imag()->compute_second_transform(icoeff[index]));

  dffull_[1] = df->get_imag()->compute_second_transform(rcoeff[index]);
  dffull_[1]->ax_plus_y( 1.0, df->get_real()->compute_second_transform(icoeff[index]));

}


RelDFFull::RelDFFull(array<shared_ptr<DFFullDist>,2> a, pair<int,int> cartesian, vector<shared_ptr<const SpinorInfo>> basis) : RelDFBase(cartesian) {
  basis_ = basis;
  dffull_ = a;
}


RelDFFull::RelDFFull(const RelDFFull& o) : RelDFBase(o.cartesian_) {
  basis_ = o.basis_;
  dffull_[0] = o.dffull_[0]->copy();
  dffull_[1] = o.dffull_[1]->copy();
}


shared_ptr<RelDFFull> RelDFFull::apply_J() const {
  array<shared_ptr<DFFullDist>,2> a{{dffull_[0]->apply_J(), dffull_[1]->apply_J()}};
  return make_shared<RelDFFull>(a, cartesian_, basis_);
}


shared_ptr<RelDFFull> RelDFFull::apply_JJ() const {
  array<shared_ptr<DFFullDist>,2> a{{dffull_[0]->apply_JJ(), dffull_[1]->apply_JJ()}};
  return make_shared<RelDFFull>(a, cartesian_, basis_);
}


shared_ptr<RelDFFull> RelDFFull::clone() const {
  array<shared_ptr<DFFullDist>,2> a{{dffull_[0]->clone(), dffull_[1]->clone()}};
  return make_shared<RelDFFull>(a, cartesian_, basis_);
}


void RelDFFull::add_product(shared_ptr<const RelDFFull> o, const shared_ptr<const ZMatrix> a, const int nocc, const int offset) {
  shared_ptr<const Matrix> ra = a->get_real_part();
  shared_ptr<const Matrix> ia = a->get_real_part();
  // taking the complex conjugate of "o"
  dffull_[0]->add_product(o->dffull_[0], ra, nocc, offset, 1.0);
  dffull_[0]->add_product(o->dffull_[1], ia, nocc, offset, 1.0);
  dffull_[1]->add_product(o->dffull_[0], ra, nocc, offset, 1.0);
  dffull_[1]->add_product(o->dffull_[1], ra, nocc, offset, -1.0);
}


void RelDFFull::ax_plus_y(complex<double> a, const RelDFFull& o) {
  if (imag(a) == 0.0) {
    const double fac = real(a);
    dffull_[0]->ax_plus_y(fac, o.dffull_[0]);
    dffull_[1]->ax_plus_y(fac, o.dffull_[1]);
  } else if (real(a) == 0.0) {
    const double fac = imag(a);
    dffull_[0]->ax_plus_y(-fac, o.dffull_[1]);
    dffull_[1]->ax_plus_y( fac, o.dffull_[0]);
  } else {
    const double rfac = real(a);
    dffull_[0]->ax_plus_y(rfac, o.dffull_[0]);
    dffull_[1]->ax_plus_y(rfac, o.dffull_[1]);
    const double ifac = imag(a);
    dffull_[0]->ax_plus_y(-ifac, o.dffull_[1]);
    dffull_[1]->ax_plus_y( ifac, o.dffull_[0]);
  }
}


void RelDFFull::scale(complex<double> a) {
  if (imag(a) == 0.0) {
    const double fac = real(a);
    dffull_[0]->scale(fac);
    dffull_[1]->scale(fac);
  } else if (real(a) == 0.0) {
    const double fac = imag(a);
    dffull_[0]->scale( fac);
    dffull_[1]->scale(-fac);
    swap(dffull_[0], dffull_[1]);
  } else {
    throw logic_error("should not happen..");
  }
}


list<shared_ptr<RelDFHalfB>> RelDFFull::back_transform(array<shared_ptr<const Matrix>,4> rcoeff, array<shared_ptr<const Matrix>,4> icoeff) const {
  list<shared_ptr<RelDFHalfB>> out;
  assert(basis_.size() == 1);
  const int alpha = basis_[0]->alpha_comp();

  for (int i = 0; i != 4; ++i) {
    // Note that icoeff should be scaled by -1.0 !!

    shared_ptr<DFHalfDist> real = dffull_[0]->back_transform(rcoeff[i]);
    real->ax_plus_y( 1.0, dffull_[1]->back_transform(icoeff[i]));

    shared_ptr<DFHalfDist> imag = dffull_[1]->back_transform(rcoeff[i]);
    imag->ax_plus_y(-1.0, dffull_[0]->back_transform(icoeff[i]));

    out.push_back(make_shared<RelDFHalfB>(array<shared_ptr<DFHalfDist>,2>{{real, imag}}, i, alpha));
  }
  return out;
}


shared_ptr<ZMatrix> RelDFFull::form_4index_1fixed(shared_ptr<const RelDFFull> a, const double fac, const int i) const {
  const size_t size = dffull_[0]->nocc1() * dffull_[0]->nocc2() * a->dffull_[0]->nocc1();
  assert(size == dffull_[1]->nocc1() * dffull_[1]->nocc2() * a->dffull_[1]->nocc1());
  assert(size == dffull_[0]->nocc1() * dffull_[0]->nocc2() * a->dffull_[1]->nocc1());

  shared_ptr<Matrix> real = dffull_[0]->form_4index_1fixed(a->dffull_[0], fac, i);
  *real += *dffull_[1]->form_4index_1fixed(a->dffull_[1], -fac, i);

  shared_ptr<Matrix> imag = dffull_[0]->form_4index_1fixed(a->dffull_[1], fac, i);
  *imag += *dffull_[1]->form_4index_1fixed(a->dffull_[0], fac, i);

  return make_shared<ZMatrix>(*real, *imag);
}


shared_ptr<ZMatrix> RelDFFull::form_4index(shared_ptr<const RelDFFull> a, const double fac) const {
  shared_ptr<Matrix> real = dffull_[0]->form_4index(a->dffull_[0], fac);
  *real += *dffull_[1]->form_4index(a->dffull_[1], -fac);

  shared_ptr<Matrix> imag = dffull_[0]->form_4index(a->dffull_[1], fac);
  *imag += *dffull_[1]->form_4index(a->dffull_[0], fac);

  return make_shared<ZMatrix>(*real, *imag);
}


shared_ptr<ZMatrix> RelDFFull::form_2index(shared_ptr<const RelDFFull> a, const double fac, const bool conjugate_left) const {
  const double ifac = conjugate_left ? -1.0 : 1.0;

  shared_ptr<Matrix> real = dffull_[0]->form_2index(a->dffull_[0], fac);
  *real -= *dffull_[1]->form_2index(a->dffull_[1], fac*ifac);

  shared_ptr<Matrix> imag = dffull_[0]->form_2index(a->dffull_[1], fac);
  *imag += *dffull_[1]->form_2index(a->dffull_[0], fac*ifac);

  return make_shared<ZMatrix>(*real, *imag);
}


shared_ptr<RelDFFull> RelDFFull::apply_2rdm(shared_ptr<const ZRDM<2>> inp) const {
  shared_ptr<const RDM<2>> rrdm = inp->get_real_part();
  shared_ptr<const RDM<2>> irdm = inp->get_imag_part();

  shared_ptr<DFFullDist> r  =  dffull_[0]->apply_2rdm(rrdm);
  r->ax_plus_y(-1.0, dffull_[1]->apply_2rdm(irdm));

  shared_ptr<DFFullDist> i  =  dffull_[1]->apply_2rdm(rrdm);
  i->ax_plus_y( 1.0, dffull_[0]->apply_2rdm(irdm));
  return make_shared<RelDFFull>(array<shared_ptr<DFFullDist>,2>{{r, i}}, cartesian_, basis_);
}
