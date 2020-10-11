//
// BAGEL - Brilliantly Advanced General Electronic Structure Library
// Filename: dipole.h
// Copyright (C) 2012 Quantum Simulation Technologies, Inc.
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


#ifndef __SRC_PROP_MULTIPOLE_H
#define __SRC_PROP_MULTIPOLE_H

#include <src/wfn/geometry.h>

namespace bagel {

class Multipole {
  protected:
    const std::shared_ptr<const Geometry> geom_;
    const std::shared_ptr<const Matrix> den_;
    const int rank_;
    const std::string jobname_;

  public:
    Multipole(std::shared_ptr<const Geometry>, std::shared_ptr<const Matrix>, const int maxrank, const std::string jobname = "");

    std::vector<double> compute() const;
};


class Dipole : public Multipole {
  public:
    Dipole(std::shared_ptr<const Geometry> g, std::shared_ptr<const Matrix> m, const std::string jobname = "") : Multipole(g, m, 1, jobname) { }
};

}

#endif
