//
// BAGEL - Parallel electron correlation program.
// Filename: breitbatch.h
// Copyright (C) 2012 Toru Shiozaki
//
// Author: Toru Shiozaki <shiozaki@northwestern.edu>
// Maintainer: Shiozaki group
//
// This file is part of the BAGEL package.
//
// The BAGEL package is free software; you can redistribute it and\/or modify
// it under the terms of the GNU Library General Public License as published by
// the Free Software Foundation; either version 2, or (at your option)
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

#ifndef __SRC_RYSINT_BREITBATCH_H
#define __SRC_RYSINT_BREITBATCH_H

#include <src/rysint/eribatch_base.h>

namespace bagel {

class BreitBatch : public ERIBatch_base {

  protected:
#if 0
    void perform_BVRR1();
    void perform_BVRR2();
    void perform_BVRR3();
    void perform_BVRR4();
    void perform_BVRR5();
    void perform_BVRR6();
    void perform_BVRR7();
    void perform_BVRR8();
    void perform_BVRR9();
    void perform_BVRR10();
    void perform_BVRR11();
    void perform_BVRR12();
    void perform_BVRR13();
#endif

  public:

    // dummy will never used.
    BreitBatch(const std::array<std::shared_ptr<const Shell>,4>&, const double max_density, const double dummy = 0.0, const bool dum = true);

    /// compute a batch of integrals
    virtual void compute();

};

}

#endif

