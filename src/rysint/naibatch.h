//
// Newint - Parallel electron correlation program.
// Filename: naibatch.h
// Copyright (C) 2009 Toru Shiozaki
//
// Author: Toru Shiozaki <shiozaki@northwestern.edu>
// Maintainer: Shiozaki group
//
// This file is part of the Newint package (to be renamed).
//
// The Newint package is free software; you can redistribute it and\/or modify
// it under the terms of the GNU Library General Public License as published by
// the Free Software Foundation; either version 2, or (at your option)
// any later version.
//
// The Newint package is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Library General Public License for more details.
//
// You should have received a copy of the GNU Library General Public License
// along with the Newint package; see COPYING.  If not, write to
// the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
//

#ifndef __src_rysint_naiprim_h
#define __src_rysint_naiprim_h 

#include <cassert>
#include <vector>
#include <src/scf/geometry.h>
#include <src/rysint/int2d.h>
#include <src/rysint/rysint.h>
#include <memory>

class NAIBatch : public RysInt {

  protected:
    std::shared_ptr<Geometry> geom_;
    int natom_;

    bool swap01_;

    double *p_;
    double *xp_, *coeff_;
    double *T_;
    double AB_[3];
    unsigned int contsize_, primsize_, size_alloc_; 
    int prim0size_, prim1size_;
    int cont0size_, cont1size_;

    int asize_, amax_, amin_, amax1_, asize_final_, asize_intermediate_;
    int amapping_[ANG_VRR_END * ANG_VRR_END * ANG_VRR_END];

    /// buffer and intermediate storage
    double *buff_;
    double *bkup_;

    /// for periodic calculations
    const int L_;
    const double A_;

    // init function
    void root_weight();

  public:
    
    NAIBatch(const std::vector<std::shared_ptr<Shell> >, const std::shared_ptr<Geometry>, const int L = 0, const double A = 0.0);
    ~NAIBatch();

    /// compute a batch of integrals
    void compute();

};

#endif

