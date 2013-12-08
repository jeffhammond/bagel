//
// BAGEL - Parallel electron correlation program.
// Filename: dfblock.cc
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

#include <src/util/taskqueue.h>
#include <src/df/dfblock.h>
#include <src/integral/libint/libint.h>
#include <src/integral/rys/eribatch.h>
#include <src/util/constants.h>
#include <src/util/simple.h>

using namespace bagel;
using namespace std;


DFBlock::DFBlock(std::shared_ptr<const StaticDist> adist_shell, std::shared_ptr<const StaticDist> adist,
                 const size_t a, const size_t b1, const size_t b2, const int as, const int b1s, const int b2s, const bool averaged)
 : btas::Tensor<double,CblasColMajor>(max(adist_shell->size(mpi__->rank()), max(adist->size(mpi__->rank()), a)), b1, b2),
   adist_shell_(adist_shell), adist_(adist), averaged_(averaged), asize_(a), b1size_(b1), b2size_(b2), astart_(as), b1start_(b1s), b2start_(b2s) {

  assert(asize_ == adist_shell->size(mpi__->rank()) || asize_ == adist_->size(mpi__->rank()) || asize_ == adist_->nele());
}


DFBlock::DFBlock(const DFBlock& o)
 : btas::Tensor<double,CblasColMajor>(max(o.adist_shell_->size(mpi__->rank()), max(o.adist_->size(mpi__->rank()), o.asize_)), o.b1size_, o.b2size_),
   adist_shell_(o.adist_shell_), adist_(o.adist_), averaged_(o.averaged_), asize_(o.asize_), b1size_(o.b1size_), b2size_(o.b2size_),
   astart_(o.astart_), b1start_(o.b1start_), b2start_(o.b2start_) {

  btas::Tensor<double,CblasColMajor>::operator=(o);
}


void DFBlock::average() {
  if (averaged_) return;
  averaged_ = true;

  // first make a send and receive buffer
  const size_t o_start = astart_;
  const size_t o_end   = o_start + asize_;
  const int myrank = mpi__->rank();
  size_t t_start, t_end;
  tie(t_start, t_end) = adist_->range(myrank);

  assert(o_end - t_end >= 0);
  assert(o_start - t_start >= 0);

  // TODO so far I am not considering the cases when data must be sent to the next neighbor; CAUTION
  const size_t asendsize = o_end - t_end;
  const size_t arecvsize = o_start - t_start;

  assert(asendsize < t_end-t_start && arecvsize < t_end-t_start);

  unique_ptr<double[]> sendbuf;
  unique_ptr<double[]> recvbuf;
  int sendtag = 0;
  int recvtag = 0;

  if (asendsize) {
    TaskQueue<CopyBlockTask> task(b2size_);

    sendbuf = unique_ptr<double[]>(new double[asendsize*b1size_*b2size_]);
    const size_t retsize = asize_ - asendsize;
    for (size_t b2 = 0, i = 0; b2 != b2size_; ++b2)
      task.emplace_back(data()+retsize+asize_*b1size_*b2, asize_, sendbuf.get()+asendsize*b1size_*b2, asendsize, asendsize, b1size_);

    task.compute();

    // send to the next node
    sendtag = mpi__->request_send(sendbuf.get(), asendsize*b1size_*b2size_, myrank+1, myrank);
  }

  if (arecvsize) {
    recvbuf = unique_ptr<double[]>(new double[arecvsize*b1size_*b2size_]);
    // recv from the previous node
    recvtag = mpi__->request_recv(recvbuf.get(), arecvsize*b1size_*b2size_, myrank-1, myrank-1);
  }

  // second move local data
  if (arecvsize || asendsize) {
    const size_t t_size = t_end - t_start;
    const size_t retsize = asize_ - asendsize;
    if (t_size <= asize_) {
      for (size_t i = 0; i != b1size_*b2size_; ++i) {
        if (i*asize_ < (i+1)*t_size-retsize) {
          copy_backward(data()+i*asize_, data()+i*asize_+retsize, data()+(i+1)*t_size);
        } else if (i*asize_ > (i+1)*t_size-retsize) {
          copy_n(data()+i*asize_, retsize, data()+(i+1)*t_size-retsize);
        }
      }
    } else {
      for (long long int i = b1size_*b2size_-1; i >= 0; --i) {
        assert(i*asize_ < (i+1)*t_size-retsize);
        copy_backward(data()+i*asize_, data()+i*asize_+retsize, data()+(i+1)*t_size);
      }
    }
  }

  // set new astart_ and asize_
  asize_ = t_end - t_start;
  astart_ = t_start;

  // set received data
  if (arecvsize) {
    // wait for recv communication
    mpi__->wait(recvtag);

    TaskQueue<CopyBlockTask> task(b2size_);
    for (size_t b2 = 0; b2 != b2size_; ++b2)
      task.emplace_back(recvbuf.get()+arecvsize*b1size_*b2, arecvsize, data()+asize_*b1size_*b2, asize_, arecvsize, b1size_);
    task.compute();
  }

  // wait for send communication
  if (asendsize) mpi__->wait(sendtag);

}


// reverse operation of average() function
void DFBlock::shell_boundary() {
  if (!averaged_) return;
  averaged_ = false;
  const size_t o_start = astart_;
  const size_t o_end = o_start + asize_;
  const int myrank = mpi__->rank();
  size_t t_start, t_end;
  tie(t_start, t_end) = adist_shell_->range(myrank);

  const size_t asendsize = t_start - o_start;
  const size_t arecvsize = t_end - o_end;
  assert(t_start >= o_start && t_end >= o_end);

  unique_ptr<double[]> sendbuf, recvbuf;
  int sendtag = 0;
  int recvtag = 0;

  if (asendsize) {
    TaskQueue<CopyBlockTask> task(b2size_);
    sendbuf = unique_ptr<double[]>(new double[asendsize*b1size_*b2size_]);
    for (size_t b2 = 0, i = 0; b2 != b2size_; ++b2)
      task.emplace_back(data()+asize_*b1size_*b2, asize_, sendbuf.get()+asendsize*b1size_*b2, asendsize, asendsize, b1size_);

    task.compute();
    assert(myrank > 0);
    sendtag = mpi__->request_send(sendbuf.get(), asendsize*b1size_*b2size_, myrank-1, myrank);
  }
  if (arecvsize) {
    assert(myrank+1 < mpi__->size());
    recvbuf = unique_ptr<double[]>(new double[arecvsize*b1size_*b2size_]);
    recvtag = mpi__->request_recv(recvbuf.get(), arecvsize*b1size_*b2size_, myrank+1, myrank+1);
  }

  if (arecvsize || asendsize) {
    const size_t t_size = t_end - t_start;
    const size_t retsize = asize_ - asendsize;
    assert(t_size >= retsize);
    if (t_size <= asize_) {
      for (size_t i = 0; i != b1size_*b2size_; ++i) {
        assert(i*asize_+asendsize > i*t_size);
        copy_n(data()+i*asize_+asendsize, retsize, data()+i*t_size);
      }
    } else {
      for (long long int i = b1size_*b2size_-1; i >= 0; --i) {
        if (i*asize_+asendsize > i*t_size) {
          copy_n(data()+i*asize_+asendsize, retsize, data()+i*t_size);
        } else if (i*asize_+asendsize < i*t_size) {
          copy_backward(data()+i*asize_+asendsize, data()+(i+1)*asize_, data()+i*t_size+retsize);
        }
      }
    }
  }

  // set new astart_ and asize_
  asize_ = t_end - t_start;
  astart_ = t_start;

  // set received data
  if (arecvsize) {
    // wait for recv communication
    mpi__->wait(recvtag);

    TaskQueue<CopyBlockTask> task(b2size_);
    for (size_t b2 = 0; b2 != b2size_; ++b2)
      task.emplace_back(recvbuf.get()+arecvsize*b1size_*b2, arecvsize, data()+asize_*b1size_*b2+(asize_-arecvsize), asize_, arecvsize, b1size_);
    task.compute();
  }

  // wait for send communication
  if (asendsize) mpi__->wait(sendtag);
}


shared_ptr<DFBlock> DFBlock::transform_second(std::shared_ptr<const Matrix> cmat, const bool trans) const {
  assert(trans ? cmat->mdim() : cmat->ndim() == b1size_);
  const double* const c = cmat->data();
  const int nocc = trans ? cmat->ndim() : cmat->mdim();

  // so far I only consider the following case
  assert(b1start_ == 0);
  auto out = make_shared<DFBlock>(adist_shell_, adist_, asize_, nocc, b2size_, astart_, 0, b2start_, averaged_);

  for (size_t i = 0; i != b2size_; ++i) {
    if (!trans)
      dgemm_("N", "N", asize_, nocc, b1size_, 1.0, data()+i*asize_*b1size_, asize_, c, b1size_, 0.0, out->data()+i*asize_*nocc, asize_);
    else
      dgemm_("N", "T", asize_, nocc, b1size_, 1.0, data()+i*asize_*b1size_, asize_, c, nocc, 0.0, out->data()+i*asize_*nocc, asize_);
  }
  return out;
}


shared_ptr<DFBlock> DFBlock::transform_third(std::shared_ptr<const Matrix> cmat, const bool trans) const {
  assert(trans ? cmat->mdim() : cmat->ndim() == b2size_);
  const double* const c = cmat->data();
  const int nocc = trans ? cmat->ndim() : cmat->mdim();

  // so far I only consider the following case
  assert(b2start_ == 0);
  auto out = make_shared<DFBlock>(adist_shell_, adist_, asize_, b1size_, nocc, astart_, b1start_, 0, averaged_);

  if (!trans)
    dgemm_("N", "N", asize_*b1size_, nocc, b2size_, 1.0, data(), asize_*b1size_, c, b2size_, 0.0, out->data(), asize_*b1size_);
  else  // trans -> back transform
    dgemm_("N", "T", asize_*b1size_, nocc, b2size_, 1.0, data(), asize_*b1size_, c, nocc, 0.0, out->data(), asize_*b1size_);

  return out;
}


shared_ptr<DFBlock> DFBlock::clone() const {
  auto out = make_shared<DFBlock>(adist_shell_, adist_, asize_, b1size_, b2size_, astart_, b1start_, b2start_, averaged_);
  out->zero();
  return out;
}


shared_ptr<DFBlock> DFBlock::copy() const {
  return make_shared<DFBlock>(*this);
}


DFBlock& DFBlock::operator+=(const DFBlock& o) { ax_plus_y( 1.0, o); return *this; }
DFBlock& DFBlock::operator-=(const DFBlock& o) { ax_plus_y(-1.0, o); return *this; }


void DFBlock::ax_plus_y(const double a, const DFBlock& o) {
  btas::axpy(a, o, *this);
}


void DFBlock::scale(const double a) {
  for (auto& i : *this) i *= a;
}


void DFBlock::add_direct_product(const shared_ptr<const Matrix> a, const shared_ptr<const Matrix> b, const double fac) {
  assert(asize_ == a->ndim() && b1size_*b2size_ == b->size());
  dger_(asize_, b1size_*b2size_, fac, a->data(), 1, b->data(), 1, data(), asize_);
}


void DFBlock::symmetrize() {
  if (b1size_ != b2size_) throw logic_error("illegal call of DFBlock::symmetrize()");
  const int n = b1size_;
  for (int i = 0; i != n; ++i)
    for (int j = i; j != n; ++j) {
      daxpy_(asize_, 1.0, data()+asize_*(j+n*i), 1, data()+asize_*(i+n*j), 1);
      copy_n(data()+asize_*(i+n*j), asize_, data()+asize_*(j+n*i));
    }
}


shared_ptr<DFBlock> DFBlock::swap() const {
  auto out = make_shared<DFBlock>(adist_shell_, adist_, asize_, b2size_, b1size_, astart_, b2start_, b1start_, averaged_);
  for (size_t b2 = b2start_; b2 != b2start_+b2size_; ++b2)
    for (size_t b1 = b1start_; b1 != b1start_+b1size_; ++b1)
      copy_n(data()+asize_*(b1+b1size_*b2), asize_, out->data()+asize_*(b2+b2size_*b1));
  return out;
}


shared_ptr<DFBlock> DFBlock::apply_rhf_2RDM(const double scale_exch) const {
  assert(b1size_ == b2size_);
  const int nocc = b1size_;
  shared_ptr<DFBlock> out = clone();
  out->zero();
  // exchange contributions
  out->ax_plus_y(-2.0*scale_exch, *this);
  // coulomb contributions (diagonal to diagonal)
  unique_ptr<double[]> diagsum(new double[asize_]);
  fill_n(diagsum.get(), asize_, 0.0);
  for (int i = 0; i != nocc; ++i)
    daxpy_(asize_, 1.0, data()+asize_*(i+nocc*i), 1, diagsum.get(), 1);
  for (int i = 0; i != nocc; ++i)
    daxpy_(asize_, 4.0, diagsum.get(), 1, out->data()+asize_*(i+nocc*i), 1);
  return out;
}


// Caution
//   o strictly assuming that we are using natural orbitals.
//
shared_ptr<DFBlock> DFBlock::apply_uhf_2RDM(std::shared_ptr<const btas::Tensor<double,CblasColMajor>> amat, std::shared_ptr<const btas::Tensor<double,CblasColMajor>> bmat) const {
  assert(b1size_ == b2size_);
  const int nocc = b1size_;
  shared_ptr<DFBlock> out = clone();
  {
    unique_ptr<double[]> d2(new double[size()]);
    // exchange contributions
    dgemm_("N", "N", asize_*nocc, nocc, nocc, 1.0, data(), asize_*nocc, amat->data(), nocc, 0.0, d2.get(), asize_*nocc);
    for (int i = 0; i != nocc; ++i)
      dgemm_("N", "N", asize_, nocc, nocc, -1.0, d2.get()+asize_*nocc*i, asize_, amat->data(), nocc, 0.0, out->data()+asize_*nocc*i, asize_);
    dgemm_("N", "N", asize_*nocc, nocc, nocc, 1.0, data(), asize_*nocc, bmat->data(), nocc, 0.0, d2.get(), asize_*nocc);
    for (int i = 0; i != nocc; ++i)
      dgemm_("N", "N", asize_, nocc, nocc, -1.0, d2.get()+asize_*nocc*i, asize_, bmat->data(), nocc, 1.0, out->data()+asize_*nocc*i, asize_);
  }

  unique_ptr<double[]> sum(new double[nocc]);
  for (int i = 0; i != nocc; ++i) sum[i] = (*amat)(i,i) + (*bmat)(i,i);
  // coulomb contributions (diagonal to diagonal)
  unique_ptr<double[]> diagsum(new double[asize_]);
  fill_n(diagsum.get(), asize_, 0.0);
  for (int i = 0; i != nocc; ++i)
    daxpy_(asize_, sum[i], data()+asize_*(i+nocc*i), 1, diagsum.get(), 1);
  for (int i = 0; i != nocc; ++i)
    daxpy_(asize_, sum[i], diagsum.get(), 1, out->data()+asize_*(i+nocc*i), 1);
  return out;
}



shared_ptr<DFBlock> DFBlock::apply_2RDM(std::shared_ptr<const btas::Tensor<double,CblasColMajor>> rdm, std::shared_ptr<const btas::Tensor<double,CblasColMajor>> rdm1, const int nclosed, const int nact) const {
  assert(nclosed+nact == b1size_ && b1size_ == b2size_);
  // checking if natural orbitals...
  {
    const double a = ddot_(nact*nact, rdm1->data(), 1, rdm1->data(), 1);
    double sum = 0.0;
    for (int i = 0; i != nact; ++i) sum += (*rdm1)(i,i)*(*rdm1)(i,i);
    if (fabs(a-sum) > numerical_zero__*100) {
      stringstream ss; ss << "DFFullDist::apply_2rdm should be called with natural orbitals " << scientific << setprecision(3) << fabs(a-sum) - numerical_zero__;
      throw logic_error(ss.str());
    }
  }
  shared_ptr<DFBlock> out = clone();
  out->zero();
  // closed-closed part
  // exchange contribution
  for (int i = 0; i != nclosed; ++i)
    for (int j = 0; j != nclosed; ++j)
      daxpy_(asize_, -2.0, data()+asize_*(j+b1size_*i), 1, out->data()+asize_*(j+b1size_*i), 1);
  // coulomb contribution
  unique_ptr<double[]> diagsum(new double[asize_]);
  fill_n(diagsum.get(), asize_, 0.0);
  for (int i = 0; i != nclosed; ++i)
    daxpy_(asize_, 1.0, data()+asize_*(i+b1size_*i), 1, diagsum.get(), 1);
  for (int i = 0; i != nclosed; ++i)
    daxpy_(asize_, 4.0, diagsum.get(), 1, out->data()+asize_*(i+b1size_*i), 1);

  // act-act part
  // compress
  unique_ptr<double[]> buf(new double[nact*nact*asize_]);
  unique_ptr<double[]> buf2(new double[nact*nact*asize_]);
  for (int i = 0; i != nact; ++i)
    for (int j = 0; j != nact; ++j)
      copy_n(data()+asize_*(j+nclosed+b1size_*(i+nclosed)), asize_, buf.get()+asize_*(j+nact*i));
  // multiply
  dgemm_("N", "N", asize_, nact*nact, nact*nact, 1.0, buf.get(), asize_, rdm->data(), nact*nact, 0.0, buf2.get(), asize_);
  // slot in
  for (int i = 0; i != nact; ++i)
    for (int j = 0; j != nact; ++j)
      copy_n(buf2.get()+asize_*(j+nact*i), asize_, out->data()+asize_*(j+nclosed+b1size_*(i+nclosed)));

  // closed-act part
  // coulomb contribution G^ia_ia = 2*gamma_ab
  // ASSUMING natural orbitals
  for (int i = 0; i != nact; ++i)
    daxpy_(asize_, 2.0*(*rdm1)(i,i), diagsum.get(), 1, out->data()+asize_*(i+nclosed+b1size_*(i+nclosed)), 1);
  unique_ptr<double[]> diagsum2(new double[asize_]);
  dgemv_("N", asize_, nact*nact, 1.0, buf.get(), asize_, rdm1->data(), 1, 0.0, diagsum2.get(), 1);
  for (int i = 0; i != nclosed; ++i)
    daxpy_(asize_, 2.0, diagsum2.get(), 1, out->data()+asize_*(i+b1size_*i), 1);
  // exchange contribution
  for (int i = 0; i != nact; ++i) {
    for (int j = 0; j != nclosed; ++j) {
      daxpy_(asize_, -(*rdm1)(i,i), data()+asize_*(j+b1size_*(i+nclosed)), 1, out->data()+asize_*(j+b1size_*(i+nclosed)), 1);
      daxpy_(asize_, -(*rdm1)(i,i), data()+asize_*(i+nclosed+b1size_*j), 1, out->data()+asize_*(i+nclosed+b1size_*j), 1);
    }
  }
  return out;
}


shared_ptr<DFBlock> DFBlock::apply_2RDM(std::shared_ptr<const btas::Tensor<double,CblasColMajor>> rdm) const {
  shared_ptr<DFBlock> out = clone();
  dgemm_("N", "T", asize_, b1size_*b2size_, b1size_*b2size_, 1.0, data(), asize_, rdm->data(), b1size_*b2size_, 0.0, out->data(), asize_);
  return out;
}


shared_ptr<Matrix> DFBlock::form_2index(const shared_ptr<const DFBlock> o, const double a) const {
  if (asize_ != o->asize_ || (b1size_ != o->b1size_ && b2size_ != o->b2size_)) throw logic_error("illegal call of DFBlock::form_2index");
  shared_ptr<Matrix> target;

  if (b1size_ == o->b1size_) {
    target = make_shared<Matrix>(b2size_,o->b2size_);
    dgemm_("T", "N", b2size_, o->b2size_, asize_*b1size_, a, data(), asize_*b1size_, o->data(), asize_*b1size_, 0.0, target->data(), b2size_);
  } else {
    assert(b2size_ == o->b2size_);
    target = make_shared<Matrix>(b1size_,o->b1size_);
    for (int i = 0; i != b2size_; ++i)
      dgemm_("T", "N", b1size_, o->b1size_, asize_, a, data()+i*asize_*b1size_, asize_, o->data()+i*asize_*o->b1size_, asize_, 1.0, target->data(), b1size_);
  }

  return target;
}


shared_ptr<Matrix> DFBlock::form_4index(const shared_ptr<const DFBlock> o, const double a) const {
  if (asize_ != o->asize_) throw logic_error("illegal call of DFBlock::form_4index");
  auto target = make_shared<Matrix>(b1size_*b2size_, o->b1size_*o->b2size_);
  dgemm_("T", "N", b1size_*b2size_, o->b1size_*o->b2size_, asize_, a, data(), asize_, o->data(), asize_, 0.0, target->data(), b1size_*b2size_);
  return target;
}


// slowest index of o is fixed to n
shared_ptr<Matrix> DFBlock::form_4index_1fixed(const shared_ptr<const DFBlock> o, const double a, const size_t n) const {
  if (asize_ != o->asize_) throw logic_error("illegal call of DFBlock::form_4index_1fixed");
  auto target = make_shared<Matrix>(b2size_*b1size_, o->b1size_);
  dgemm_("T", "N", b1size_*b2size_, o->b1size_, asize_, a, data(), asize_, o->data()+n*asize_*o->b1size_, asize_, 0.0, target->data(), b1size_*b2size_);
  return target;
}


shared_ptr<Matrix> DFBlock::form_aux_2index(const shared_ptr<const DFBlock> o, const double a) const {
  if (b1size_ != o->b1size_ || b2size_ != o->b2size_) throw logic_error("illegal call of DFBlock::form_aux_2index");
  auto target = make_shared<Matrix>(asize_, o->asize_);
  dgemm_("N", "T", asize_, o->asize_, b1size_*b2size_, a, data(), asize_, o->data(), o->asize_, 0.0, target->data(), asize_);
  return target;
}


unique_ptr<double[]> DFBlock::form_vec(const shared_ptr<const Matrix> den) const {
  unique_ptr<double[]> out(new double[asize_]);
  assert(den->ndim() == b1size_ && den->mdim() == b2size_);
  dgemv_("N", asize_, b1size_*b2size_, 1.0, data(), asize_, den->data(), 1, 0.0, out.get(), 1);
  return out;
}


shared_ptr<Matrix> DFBlock::form_mat(const double* fit) const {
  auto out = make_shared<Matrix>(b1size_,b2size_);
  dgemv_("T", asize_, b1size_*b2size_, 1.0, data(), asize_, fit, 1, 0.0, out->data(), 1);
  return out;
}


void DFBlock::contrib_apply_J(const shared_ptr<const DFBlock> o, const shared_ptr<const Matrix> d) {
  if (b1size_ != o->b1size_ || b2size_ != o->b2size_) throw logic_error("illegal call of DFBlock::contrib_apply_J");
  dgemm_("N", "N", asize_, b1size_*b2size_, o->asize_, 1.0, d->element_ptr(astart_, o->astart_), d->ndim(), o->data(), o->asize_,
                                                        1.0, data(), asize_);
}


void DFBlock::copy_block(const std::shared_ptr<const Matrix> o, const int jdim, const size_t offset) {
  assert(o->size() == asize_*jdim);
  copy_n(o->data(), asize_*jdim, data()+offset);
}


void DFBlock::add_block(const std::shared_ptr<const Matrix> o, const int jdim, const size_t offset, const double fac) {
  assert(o->size() == asize_*jdim);
  daxpy_(asize_*jdim, fac, o->data(), 1, data()+offset, 1);
}


shared_ptr<Matrix> DFBlock::form_Dj(const shared_ptr<const Matrix> o, const int jdim) const {
  assert(o->size() == b1size_*b2size_*jdim);
  auto out = make_shared<Matrix>(asize_, jdim);
  dgemm_("N", "N", asize_, jdim, b1size_*b2size_, 1.0, data(), asize_, o->data(), b1size_*b2size_, 0.0, out->data(), asize_);
  return out;
}


shared_ptr<Matrix> DFBlock::get_block(const int ist, const int i, const int jst, const int j, const int kst, const int k) const {
  const int ista = ist - astart_;
  const int jsta = jst - b1start_;
  const int ksta = kst - b2start_;
  const int ifen = ist + i - astart_;
  const int jfen = jst + j - b1start_;
  const int kfen = kst + k - b2start_;
  if (ista < 0 || jsta < 0 || ksta < 0 || ifen > asize_ || jfen > b1size_ || kfen > b2size_)
    throw logic_error("illegal call of DFBlock::get_block");

  // TODO we need 3-index tensor class here!
  auto out = make_shared<Matrix>(i, j*k);
  double* d = out->data();
  for (int kk = ksta; kk != kfen; ++kk)
    for (int jj = jsta; jj != jfen; ++jj, d += i)
      copy_n(data()+ista+asize_*(jj+b1size_*kk), i, d);

  return out;
}
