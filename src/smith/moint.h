//
// BAGEL - Parallel electron correlation program.
// Filename: moint.h
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

// A base class for electorn Correlation methods
// Certain kinds of MO integrals are formed.
//   - aaii (assumes DF - TODO half transformed DF vector might be available..)
//

#ifndef __SRC_SMITH_MOINT_H
#define __SRC_SMITH_MOINT_H

#include <stddef.h>
#include <memory>
#include <stdexcept>
#include <src/math/algo.h>
#include <src/wfn/reference.h>
#include <src/smith/tensor.h>
#include <src/scf/fock.h>

namespace bagel {
namespace SMITH {

// the template parameter T specifies the storage type

template <typename T>
class K2ext {
  protected:
    std::shared_ptr<const Reference> ref_;
    std::shared_ptr<const Coeff> coeff_;
    std::vector<IndexRange> blocks_;
    std::shared_ptr<Tensor<T>> data_;

    // some handwritten drivers
    std::map<size_t, std::shared_ptr<DFFullDist>> generate_list() {
      std::shared_ptr<const DFDist> df = ref_->geom()->df();

      // It is the easiest to do integral transformation for each blocks.
      assert(blocks_.size() == 4);
      std::map<size_t, std::shared_ptr<DFFullDist>> dflist;
      // AO dimension
      const size_t nbasis = df->nbasis0();
      assert(df->nbasis0() == df->nbasis1());

      // occ loop
      for (auto& i0 : blocks_[0]) {
        std::shared_ptr<DFHalfDist> df_half = df->compute_half_transform(coeff_->slice(i0.offset(), i0.offset()+i0.size()))->apply_J();
        // virtual loop
        for (auto& i1 : blocks_[1]) {
          std::shared_ptr<DFFullDist> df_full = df_half->compute_second_transform(coeff_->slice(i1.offset(), i1.offset()+i1.size()));
          dflist.insert(make_pair(generate_hash_key(i0, i1), df_full));
        }
      }
      return dflist;
    }

    void form_4index(const std::map<size_t, std::shared_ptr<DFFullDist>>& dflist) {
      // form four-index integrals
      // TODO this part should be heavily parallelized
      // TODO i01 < i23 symmetry should be used.
      for (auto& i0 : blocks_[0]) {
        for (auto& i1 : blocks_[1]) {
          // find three-index integrals
          auto iter01 = dflist.find(generate_hash_key(i0, i1));
          assert(iter01 != dflist.end());
          std::shared_ptr<DFFullDist> df01 = iter01->second;
          size_t hashkey01 = generate_hash_key(i0, i1);

          for (auto& i2 : blocks_[2]) {
            for (auto& i3 : blocks_[3]) {
              // find three-index integrals
              size_t hashkey23 = generate_hash_key(i2, i3);
              if (hashkey23 > hashkey01) continue;

              auto iter23 = dflist.find(generate_hash_key(i2, i3));
              assert(iter23 != dflist.end());
              std::shared_ptr<const DFFullDist> df23 = iter23->second;

              // contract
              // TODO form_4index function now generates global 4 index tensor. This should be localized.
              std::shared_ptr<Matrix> tmp = df01->form_4index(df23, 1.0);
              std::unique_ptr<double[]> target(new double[tmp->size()]);
              std::copy_n(tmp->data(), tmp->size(), target.get()); // unnecessary copy

              // move in place
              if (hashkey23 != hashkey01) {
                std::unique_ptr<double[]> target2(new double[i0.size()*i1.size()*i2.size()*i3.size()]);
                blas::transpose(target.get(), i0.size()*i1.size(), i2.size()*i3.size(), target2.get());

                data_->put_block(target2, i2, i3, i0, i1);
              }

              data_->put_block(target, i0, i1, i2, i3);
            }
          }
        }
      }
    }

  public:
    K2ext(std::shared_ptr<const Reference> r, std::shared_ptr<const Coeff> c, std::vector<IndexRange> b) : ref_(r), coeff_(c), blocks_(b) {
      // so far MOInt can be called for 2-external K integral and all-internals.
      if (blocks_[0] != blocks_[2] || blocks_[1] != blocks_[3])
        throw std::logic_error("MOInt called with wrong blocks");
      data_ = std::shared_ptr<Tensor<T>>(new Tensor<T>(blocks_, false));
      form_4index(generate_list());
    }

    ~K2ext() {}

    std::shared_ptr<Tensor<T>> data() { return data_; }
    std::shared_ptr<Tensor<T>> tensor() { return data_; }

};


template <typename T>
class MOFock {
  protected:
    std::shared_ptr<const Reference> ref_;
    std::shared_ptr<Coeff> coeff_;
    std::vector<IndexRange> blocks_;
    std::shared_ptr<Tensor<T>> data_;
    std::shared_ptr<Tensor<T>> hcore_;

  public:
    MOFock(std::shared_ptr<const Reference> r, std::vector<IndexRange> b) : ref_(r), coeff_(new Coeff(*ref_->coeff())), blocks_(b) {
      // for simplicity, I assume that the Fock matrix is formed at once (may not be needed).
      assert(b.size() == 2 && b[0] == b[1]);

      data_  = std::shared_ptr<Tensor<T>>(new Tensor<T>(blocks_, false));
      hcore_ = std::shared_ptr<Tensor<T>>(new Tensor<T>(blocks_, false));

      std::shared_ptr<const Matrix> hcore = ref_->hcore();

      std::shared_ptr<Matrix> den;
      if (ref_->nact() == 0) {
        den = ref_->coeff()->form_density_rhf(ref_->nclosed());
      } else {
        // TODO NOTE THAT RDM 0 IS HARDWIRED should be fixed later on
        std::shared_ptr<const Matrix> tmp = ref_->rdm1(0)->rdm1_mat(ref_->nclosed(), true);
        // slice of coeff
        std::shared_ptr<const Matrix> c = ref_->coeff()->slice(0, ref_->nocc());
        // transforming to AO basis
        den = std::shared_ptr<Matrix>(new Matrix(*c * *tmp ^ *c));
      }

      std::shared_ptr<const Matrix> fock1(new Fock<1>(ref_->geom(), hcore, den, r->schwarz()));
      const Matrix forig = *r->coeff() % *fock1 * *r->coeff();

      // if closed/virtual orbitals are present, we diagonalize the fock operator within this subspace
      const int nclosed = ref_->nclosed();
      const int nocc    = ref_->nocc();
      const int nvirt   = ref_->nvirt();
      const int nbasis  = ref_->geom()->nbasis();
      std::unique_ptr<double[]> eig(new double[nbasis]);
      if (nclosed > 1) {
        std::shared_ptr<Matrix> fcl = forig.get_submatrix(0, 0, nclosed, nclosed);
        fcl->diagonalize(eig.get());
        dgemm_("N", "N", nbasis, nclosed, nclosed, 1.0, ref_->coeff()->data(), nbasis, fcl->data(), nclosed, 0.0, coeff_->data(), nbasis);
      }
      if (nvirt > 1) {
        std::shared_ptr<Matrix> fvirt = forig.get_submatrix(nocc, nocc, nvirt, nvirt);
        fvirt->diagonalize(eig.get());
        dgemm_("N", "N", nbasis, nvirt, nvirt, 1.0, ref_->coeff()->element_ptr(0,nocc), nbasis, fvirt->data(), nvirt, 0.0, coeff_->element_ptr(0,nocc), nbasis);
      }
      const Matrix f = *coeff_ % *fock1 * *coeff_;
      const Matrix hc = *coeff_ % *hcore * *coeff_;

      for (auto& i0 : blocks_[0]) {
        for (auto& i1 : blocks_[1]) {
          {
            std::shared_ptr<Matrix> tm = f.get_submatrix(i1.offset(), i0.offset(), i1.size(), i0.size());
            std::unique_ptr<double[]> target(new double[tm->size()]);
            std::copy_n(tm->data(), tm->size(), target.get());
            data_->put_block(target, i1, i0);
          } {
            std::shared_ptr<Matrix> tm = hc.get_submatrix(i1.offset(), i0.offset(), i1.size(), i0.size());
            std::unique_ptr<double[]> target(new double[tm->size()]);
            std::copy_n(tm->data(), tm->size(), target.get());
            hcore_->put_block(target, i1, i0);
          }
        }
      }
    }
    ~MOFock() {}

    std::shared_ptr<Tensor<T>> data() { return data_; }
    std::shared_ptr<Tensor<T>> tensor() { return data_; }
    std::shared_ptr<Tensor<T>> hcore() { return hcore_; }
    std::shared_ptr<const Coeff> coeff() const { return coeff_; }
};


template <typename T>
class Ci {
  protected:
    std::shared_ptr<const Reference> ref_;
    std::vector<IndexRange> blocks_;
    std::size_t ci_size_;
    std::shared_ptr<Tensor<T>>  rdm0deriv_;


  public:
    Ci(std::shared_ptr<const Reference> r, std::vector<IndexRange> b, std::shared_ptr<const Civec> c) : ref_(r), blocks_(b), ci_size_(c->size()) {
      assert(b.size() == 1);

      // form ci coefficient tensor
      rdm0deriv_  = std::shared_ptr<Tensor<T>>(new Tensor<T>(blocks_, false));

      for (auto& i0 : blocks_[0]) {
        const size_t size = i0.size();
        std::unique_ptr<double[]> cc(new double[size]);
        int iall = 0;
        for (int j0 = i0.offset(); j0 != i0.offset()+i0.size(); ++j0, ++iall) {
          cc[iall] = c->data(j0);
        }
        rdm0deriv_->put_block(cc, i0);
      }

    }
    ~Ci() {}

    std::shared_ptr<Tensor<T>> tensor() const { return rdm0deriv_; }
};

}
}

#endif

