//
// Newint - Parallel electron correlation program.
// Filename: eribatch.cc
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

#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <src/rysint/eribatch.h>
#include <src/rysint/inline.h>
#include <src/rysint/macros.h>

using namespace std;


ERIBatch::ERIBatch(const array<shared_ptr<const Shell>,4>& _info, const double max_density, const double dummy, const bool dum)
:  ERIBatch_base(_info, max_density, 0) {
  vrr_ = shared_ptr<VRRListBase>(new VRRList());

}


ERIBatch::~ERIBatch() {
}


