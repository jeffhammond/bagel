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

template<typename DataType>
class RDM_base : public btas::Tensor<DataType> {
  protected:
    const int norb_;
    int rank_;

  public:
    RDM_base(const int n, const int rank) : btas::Tensor<DataType>(std::vector<unsigned long>(rank, n)), norb_(n), rank_(rank) {
      zero();
    }

    RDM_base(const RDM_base<DataType>& o) : btas::Tensor<DataType>(o), norb_(o.norb_), rank_(o.rank_) {
    }

    RDM_base(RDM_base<DataType>&& o) = default;

    // TODO in principle we should be able to get rid of the data() functions
    DataType* data() { return &(*this->begin()); }
    const DataType* data() const { return &(*this->begin()); }

    void zero() { std::fill(this->begin(), this->end(), static_cast<DataType>(0.0)); }
    void ax_plus_y(const DataType a, const RDM_base<DataType>& o) { btas::axpy(a, o, *this); }
    void ax_plus_y(const DataType& a, const std::shared_ptr<RDM_base<DataType>>& o) { this->ax_plus_y(a, *o); }
    void scale(const DataType& a) { std::for_each(this->begin(), this->end(), [&a](DataType& p){ p *= a; }); }

    int norb() const { return norb_; }

};


template <int rank, typename DataType = double>
class RDM : public RDM_base<DataType> {
  public:
    RDM(const int n) : RDM_base<DataType>(n, rank) { }
    RDM(const RDM<rank,DataType>& o) : RDM_base<DataType>(o) { }
    RDM(RDM<rank,DataType>&& o) : RDM_base<DataType>(std::forward(o)) { }

    std::shared_ptr<RDM<rank,DataType>> clone() const { return std::make_shared<RDM<rank,DataType>>(this->norb_); }
    std::shared_ptr<RDM<rank,DataType>> copy() const { return std::make_shared<RDM<rank,DataType>>(*this); }

    template<typename ...args>
    DataType& element(const args&... index) { return (*this)(index...); }

    template<typename ...args>
    const DataType& element(const args&... index) const { return (*this)(index...); }

    RDM<rank,DataType>& operator+=(const RDM<rank,DataType>& o) { this->ax_plus_y(1.0, o); return *this; }
    RDM<rank,DataType>& operator-=(const RDM<rank,DataType>& o) { this->ax_plus_y(-1.0, o); return *this; }
    RDM<rank,DataType> operator+(const RDM<rank,DataType>& o) const { RDM<rank,DataType> out(*this); out.ax_plus_y(1.0, o); return out; }
    RDM<rank,DataType> operator-(const RDM<rank,DataType>& o) const { RDM<rank,DataType> out(*this); out.ax_plus_y(-1.0, o); return out; }

    // returns if this is natural orbitals - only for rank 1
    bool natural_orbitals() const {
      throw std::logic_error("RDM<N>::natural_orbitals() should not be called with N>1");
      return true;
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
