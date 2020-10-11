//
// BAGEL - Brilliantly Advanced General Electronic Structure Library
// Filename: zqvec.h
// Copyright (C) 2013 Quantum Simulation Technologies, Inc.
//
// Author: Toru Shiozaki <shiozaki@qsimulate.com>
// Maintainer: QSimulate
//
// This file is part of the BAGEL package.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//


#ifndef __BAGEL_SRC_ZCASSCF_ZQVEC_H
#define __BAGEL_SRC_ZCASSCF_ZQVEC_H

#include <src/ci/zfci/zharrison.h> // 2RDM and transformed integrals

namespace bagel {

class ZQvec : public ZMatrix {
  protected:

  public:
    ZQvec(const ZMatrix& a) : ZMatrix(a) {}

    ZQvec(const int n, const int m, std::shared_ptr<const Geometry> geom, std::shared_ptr<const ZMatrix> rcoeff, std::shared_ptr<const ZMatrix> acoeff, const int nclosed,
          std::shared_ptr<const ZHarrison> fci, const bool gaunt, const bool breit);
};

}

#endif
