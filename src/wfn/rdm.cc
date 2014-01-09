//
// BAGEL - Parallel electron correlation program.
// Filename: rdm.cc
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


#include <src/wfn/rdm.h>
#include <src/math/algo.h>

using namespace bagel;
using namespace std;


template<>
bool RDM<1>::natural_orbitals() const {
  const double a = ddot_(norb_*norb_, data(), 1, data(), 1);
  const double b = ddot_(norb_, data(), norb_+1, data(), norb_+1);
  return fabs(a-b) < 1.0e-12;
}


template<>
pair<shared_ptr<Matrix>, vector<double>> RDM<1>::generate_natural_orbitals() const {
  auto buf = make_shared<Matrix>(norb_,norb_,true);
  buf->add_diag(2.0);
  btas::axpy(-1.0, *this, *buf);

  vector<double> vec(norb_);
  buf->diagonalize(vec.data());

  for (auto& i : vec) i = 2.0-i;

  map<int,int> emap;
  auto buf2 = make_shared<Matrix>(norb_,norb_,true);
  vector<double> vec2(norb_);
  // sort eigenvectors so that buf is close to a unit matrix
  // target column
  for (int i = 0; i != norb_; ++i) {
    // first find the source column
    tuple<int, double> max = make_tuple(-1, 0.0);
    for (int j = 0; j != norb_; ++j)
      if (fabs(buf->element(i,j)) > get<1>(max))
        max = make_tuple(j, fabs(buf->element(i,j)));

    // register to emap
    if (emap.find(get<0>(max)) != emap.end()) throw logic_error("this should not happen. RDM<1>::generate_natural_orbitals()");
    emap.insert(make_pair(get<0>(max), i));

    // copy to the target
    copy_n(buf->element_ptr(0,get<0>(max)), norb_, buf2->element_ptr(0,i));
    vec2[i] = vec[get<0>(max)];
  }

  // fix the phase
  for (int i = 0; i != norb_; ++i) {
    if (buf2->element(i,i) < 0.0)
      dscal_(norb_, -1.0, buf2->element_ptr(0,i), 1);
  }

  return make_pair(buf2, vec2);
}


template<>
vector<double> RDM<1>::diag() const {
  vector<double> out(this->norb_);
  for (int i = 0; i != this->norb_; ++i) out[i] = element(i,i);
  return out;
}

template<>
void RDM<1>::transform(const shared_ptr<Matrix>& coeff) {
  auto tmp = this->clone();
  btas::gemm(CblasNoTrans, CblasNoTrans, 1.0, *this, *coeff, 0.0, *tmp);
  btas::gemm(CblasTrans, CblasNoTrans, 1.0, *coeff, *tmp, 0.0, *this);
}


template<>
void RDM<2>::transform(const shared_ptr<Matrix>& coeff) {
  const int dim = norb_*norb_;
  const double* start = coeff->data();
  unique_ptr<double[]> buf(new double[dim*dim]);
  // first half transformation
  dgemm_("N", "N", dim*norb_, norb_, norb_, 1.0, data(), dim*norb_, start, norb_, 0.0, buf.get(), dim*norb_);
  for (int i = 0; i != norb_; ++i)
    dgemm_("N", "N", dim, norb_, norb_, 1.0, buf.get()+i*dim*norb_, dim, start, norb_, 0.0, data()+i*dim*norb_, dim);
  // then tranpose
  blas::transpose(data(), dim, dim, buf.get());
  // and do it again
  dgemm_("N", "N", dim*norb_, norb_, norb_, 1.0, buf.get(), dim*norb_, start, norb_, 0.0, data(), dim*norb_);
  for (int i = 0; i != norb_; ++i)
    dgemm_("N", "N", dim, norb_, norb_, 1.0, data()+i*dim*norb_, dim, start, norb_, 0.0, buf.get()+i*dim*norb_, dim);
  // to make sure for non-symmetric density matrices (and anyway this should be cheap).
  blas::transpose(buf.get(), dim, dim, data());
}


template<>
shared_ptr<Matrix> RDM<1>::rdm1_mat(const int nclosed, const bool all) const {
  auto out = make_shared<Matrix>(nclosed+norb_, nclosed+norb_);
  if (all)
    for (int i = 0; i != nclosed; ++i) out->element(i,i) = 2.0;
  for (int i = 0; i != norb_; ++i)
    for (int j = 0; j != norb_; ++j)
      out->element(j+nclosed, i+nclosed) = element(j,i);
  return out;
}


template<>
void RDM<1>::print(const double thresh) const {
  const double* ptr = data();
  for (int i = 0; i != norb_; ++i)
    for (int j = 0; j != norb_; ++j)
      cout << setw(12) << setprecision(7) << *ptr++ << endl;
}


template<>
void RDM<2>::print(const double thresh) const {
  const double* ptr = data();
  for (int i = 0; i != norb_; ++i)
    for (int j = 0; j != norb_; ++j)
      for (int k = 0; k != norb_; ++k)
        for (int l = 0; l != norb_; ++l, ++ptr)
          if (fabs(*ptr) > thresh)
            cout << setw(3) << l << setw(3)
                      << k << setw(3) << j << setw(3) << i
                      << setw(12) << setprecision(7) << *ptr << endl;
}


template<>
void RDM<3>::print(const double thresh) const {
  const double* ptr = data();
  for (int i = 0; i != norb_; ++i)
    for (int j = 0; j != norb_; ++j)
      for (int k = 0; k != norb_; ++k)
        for (int l = 0; l != norb_; ++l)
        for (int m = 0; m != norb_; ++m)
        for (int n = 0; n != norb_; ++n, ++ptr)
          if (fabs(*ptr) > thresh)
            cout << setw(3) << n << setw(3) << m << setw(3) << l << setw(3)
                 << k << setw(3) << j << setw(3) << i
                 << setw(12) << setprecision(7) << *ptr << endl;
}
