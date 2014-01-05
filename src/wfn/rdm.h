//
// BAGEL - Parallel electron correlation program.
// Filename: rdm.h
// Copyright (C) 2011 Toru Shiozaki
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


#ifndef __BAGEL_WFN_RDM_H
#define __BAGEL_WFN_RDM_H

#include <type_traits>
#include <src/wfn/geometry.h>
#include <src/math/matrix.h>

#include <btas/tensor.h>

namespace bagel {

template<int rank, typename DataType>
class RDM_base : public btas::TensorN<DataType,rank*2> {
  protected:
    const int norb_;

  public:
    RDM_base(const int n) : btas::TensorN<DataType,rank*2>(btas::Range(btas::Range1(n), rank*2)), norb_(n) {
      zero();
    }

    RDM_base(const RDM_base<rank, DataType>& o) : btas::TensorN<DataType,rank*2>(o), norb_(o.norb_) {
    }

    RDM_base(RDM_base<rank, DataType>&& o) = default;

    // TODO in principle we should be able to get rid of the data() functions
    DataType* data() { return &(*this->begin()); }
    const DataType* data() const { return &(*this->cbegin()); }

    using btas::TensorN<DataType, rank*2>::begin;
    using btas::TensorN<DataType, rank*2>::end;

    void zero() { std::fill(begin(), end(), static_cast<DataType>(0.0)); }
    void ax_plus_y(const DataType a, const RDM_base<rank, DataType>& o) { btas::axpy(a, o, *this); }
    void ax_plus_y(const DataType& a, const std::shared_ptr<RDM_base<rank, DataType>>& o) { this->ax_plus_y(a, *o); }
    void scale(const DataType& a) { btas::scal(a, *this); }

    int norb() const { return norb_; }

};


template <int rank, typename DataType = double>
class RDM : public RDM_base<rank, DataType> {
  public:
    RDM(const int n) : RDM_base<rank, DataType>(n) { }
    RDM(const RDM<rank,DataType>& o) : RDM_base<rank, DataType>(o) { }
    RDM(RDM<rank,DataType>&& o) : RDM_base<rank, DataType>(std::forward(o)) { }

    std::shared_ptr<RDM<rank,DataType>> clone() const { return std::make_shared<RDM<rank,DataType>>(this->norb_); }
    std::shared_ptr<RDM<rank,DataType>> copy() const { return std::make_shared<RDM<rank,DataType>>(*this); }

    template<typename ...args>
    DataType& element(const args&... index) { return (*this)(index...); }

    template<typename ...args>
    const DataType& element(const args&... index) const { return (*this)(index...); }

    // returns if this is natural orbitals - only for rank 1
    bool natural_orbitals() const {
      throw std::logic_error("RDM<N>::natural_orbitals() should not be called with N>1");
      return true;
    }

    std::shared_ptr<RDM<rank, double>> get_real_part() const {
      auto out = std::make_shared<RDM<rank, double>>(this->norb_);
      std::transform(this->begin(), this->end(), out->begin(), [](const std::complex<double>& a){ return detail::real(a); });
      return out;
    }

    std::shared_ptr<RDM<rank, double>> get_imag_part() const {
      auto out = std::make_shared<RDM<rank, double>>(this->norb_);
      std::transform(this->begin(), this->end(), out->begin(), [](const std::complex<double>& a){ return detail::imag(a); });
      return out;
    }

    std::shared_ptr<Matrix> rdm1_mat(const int nclosed, const bool all = true) const {
      throw std::logic_error("RDM<N>::rdm1_mat() should not be called with N>1");
      return std::shared_ptr<Matrix>();
    }

    std::pair<std::shared_ptr<Matrix>, std::vector<double>> generate_natural_orbitals() const {
      throw std::logic_error("RDM<N>::generate_natural_orbitals() should not be called with N>1");
      return std::pair<std::shared_ptr<Matrix>, std::vector<double>>();
    }

    void transform(const std::shared_ptr<Matrix>& coeff) { throw std::logic_error("RDM<N>::transform() (N>3) not implemented yet"); }

    std::vector<DataType> diag() const {
      throw std::logic_error("Unspecialized RDM<N>::diag() should not be called");
      return std::vector<DataType>();
    }

    void print(const double thresh = 1.0e-3) const { throw std::logic_error("RDM<N>::print() (N>3) not implemented yet"); }
};

template <int rank>
using ZRDM = RDM<rank, std::complex<double>>;

template<> bool RDM<1,double>::natural_orbitals() const;

template<> std::pair<std::shared_ptr<Matrix>, std::vector<double>> RDM<1,double>::generate_natural_orbitals() const;

template<> std::vector<double> RDM<1,double>::diag() const;

template<> void RDM<1,double>::transform(const std::shared_ptr<Matrix>& coeff);
template<> void RDM<2,double>::transform(const std::shared_ptr<Matrix>& coeff);

template<> std::shared_ptr<Matrix> RDM<1,double>::rdm1_mat(const int nclosed, const bool all) const;

template<> void RDM<1,double>::print(const double thresh) const;
template<> void RDM<2,double>::print(const double thresh) const;
template<> void RDM<3,double>::print(const double thresh) const;

}

#endif
