//
// BAGEL - Parallel electron correlation program.
// Filename: hcore.cc
// Copyright (C) 2009 Toru Shiozaki
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


#include <src/molecule/hcore.h>
#include <src/integral/os/kineticbatch.h>
#include <src/integral/os/mmbatch.h>
#include <src/integral/rys/naibatch.h>
#include <src/integral/rys/eribatch.h>
#include <src/integral/libint/libint.h>

using namespace std;
using namespace bagel;

Hcore::Hcore(const shared_ptr<const Molecule> mol) : Matrix1e(mol) {

  init();
  fill_upper();

}


void Hcore::computebatch(const array<shared_ptr<const Shell>,2>& input, const int offsetb0, const int offsetb1) {

  // input = [b1, b0]
  assert(input.size() == 2);
  const int dimb1 = input[0]->nbasis();
  const int dimb0 = input[1]->nbasis();

  {
    KineticBatch kinetic(input);
    kinetic.compute();

    copy_block(offsetb1, offsetb0, dimb1, dimb0, kinetic.data());
  }
  {
    NAIBatch nai(input, mol_);
    nai.compute();

    add_block(1.0, offsetb1, offsetb0, dimb1, dimb0, nai.data());
  }

  if (mol_->has_finite_nucleus()) {
    auto dummy = make_shared<const Shell>(input[0]->spherical());
    for (auto& i : mol_->atoms()) {
      if (i->finite_nucleus()) {
        const double fac = - i->atom_charge()*pow(i->atom_exponent()/pi__, 1.5);
        auto in = make_shared<Shell>(i->spherical(), i->position(), 0, vector<double>{i->atom_exponent()}, vector<vector<double>>{{fac}}, vector<pair<int,int>>{make_pair(0,1)});
        const array<shared_ptr<const Shell>,4> shells{{ dummy, in, input[0], input[1] }};
#ifdef LIBINT_INTERFACE
        Libint eri(shells);
#else
        ERIBatch eri(shells, 2.0);
#endif
        eri.compute();
        add_block(1.0, offsetb1, offsetb0, dimb1, dimb0, eri.data());
      }
    }
  }

  if (mol_->external()) {
    DipoleBatch dipole(input, mol_);
    dipole.compute();
    const size_t block = dipole.size_block();
    const double* dip = dipole.data();

    int cnt = 0;
    for (int i = offsetb0; i != dimb0 + offsetb0; ++i) {
      for (int j = offsetb1; j != dimb1 + offsetb1; ++j, ++cnt) {
        element(j, i) += dip[cnt        ]*mol_->external(0);
        element(j, i) += dip[cnt+block  ]*mol_->external(1);
        element(j, i) += dip[cnt+block*2]*mol_->external(2);
      }
    }
  }
}


