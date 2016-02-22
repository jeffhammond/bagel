//
// BAGEL - Brilliantly Advanced General Electronic Structure Library
// Filename: storage.cc
// Copyright (C) 2012 Toru Shiozaki
//
// Author: Toru Shiozaki <shiozaki@northwestern.edu>
// Maintainer: Shiozaki group
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

#include <bagel_config.h>
#ifdef COMPILE_SMITH

#include <ga.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <src/util/f77.h>
#include <src/util/math/algo.h>
#include <src/smith/storage.h>
#include <src/util/parallel/mpi_interface.h>

using namespace bagel::SMITH;
using namespace std;

// size contains hashkey and length (in this order)
template<typename DataType>
StorageIncore<DataType>::StorageIncore(const map<size_t, size_t>& size, bool init) : initialized_(false) {
  static_assert(is_same<DataType, double>::value or is_same<DataType, complex<double>>::value, "illegal Type in StorageIncore");

  using GA_Type = typename conditional<is_same<double,DataType>::value, double, DoubleComplex>::type;

  static_assert(sizeof(GA_Type) == sizeof(DataType),
    "SMITH assumes that the GA and BAGEL data type have the same size");

  // first prepare some variables
  totalsize_ = 0;
  for (auto& i : size) {
    if (i.second > 0)
      hashtable_.emplace(i.first, make_pair(totalsize_, totalsize_+i.second-1));
    totalsize_ += i.second;
  }

  // store block data
  const size_t blocksize = (totalsize_-1)/mpi__->size()+1;
  size_t tsize = 0;
  for (auto& i : size) {
    if (i.second == 0) continue;
    if (blocks_.size()*blocksize <= tsize)
      blocks_.push_back(tsize);
    tsize += i.second;
  }
  blocks_.push_back(totalsize_);
  assert(totalsize_ == tsize);

  // here we initialize the global array storage
  if (init)
    initialize();
}


template<typename DataType>
void StorageIncore<DataType>::initialize() {
  assert(!initialized_);
  // create MPI window
  //auto type = is_same<double,DataType>::value ? MT_C_DBL : MT_C_DCPL;
  // some implementations of MPI may not provide MPI_CXX_DOUBLE_COMPLEX, which is part of MPI-3
  // see https://github.com/solomonik/ctf/issues/10 for discussion
#if MPI_VERSION >= 3
  auto type = is_same<double,DataType>::value ? MPI_DOUBLE : MPI_CXX_DOUBLE_COMPLEX;
#else
  auto type = is_same<double,DataType>::value ? MPI_DOUBLE : MPI_C_DOUBLE_COMPLEX;
#endif
  int64_t nblocks = blocks_.size() - 1;
  assert(nblocks <= mpi__->size());
  //ga_ = NGA_Create_irreg64(type, 1, &totalsize_, const_cast<char*>(""), &nblocks, blocks_.data());
  // TODO need a hash function or something like that to map blocks to processes,
  // since that was opaque in GA but is not in MPI-3...
  MPI_Aint size;
  MPI_Info info = MPI_INFO_NULL;  // some info keys may be useful.  do this later.
  MPI_Comm comm = MPI_COMM_WORLD; // hopefully Bagel creates its own communicator.
  MPI_Win_allocate(size,1 info, comm, &win_base_, &win_);
  initialized_ = true;
  zero();
}


template<typename DataType>
StorageIncore<DataType>::~StorageIncore() {
  //GA_Destroy(ga_);
  // This deallocates memory if MPI_Win_allocate is used;
  // if we use MPI_Win_create, we need MPI_Win_free and MPI_Free_mem.
  MPI_Win_free(&win_);
}


template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block_(const size_t& key) const {
  assert(initialized_);
  // first find a key
  auto hash = hashtable_.find(key);
  if (hash == hashtable_.end())
    throw logic_error("a key was not found in Storage::get_block(const size_t&)");
  auto p = hash->second;
  int64_t size = p.second - p.first + 1;
  unique_ptr<DataType[]> out(new DataType[size]);
  NGA_Get64(ga_, &p.first, &p.second, out.get(), &size);
  // MPI_Get
  return move(out);
}


template<typename DataType>
void StorageIncore<DataType>::put_block_(const unique_ptr<DataType[]>& dat, const size_t& key) {
  assert(initialized_);
  auto hash = hashtable_.find(key);
  if (hash == hashtable_.end())
    throw logic_error("a key was not found in Storage::put_block(const size_t&)");
  auto p = hash->second;
  int64_t size = p.second - p.first + 1;
  NGA_Put64(ga_, &p.first, &p.second, dat.get(), &size);
  // MPI_Put
}


template<typename DataType>
void StorageIncore<DataType>::add_block_(const unique_ptr<DataType[]>& dat, const size_t& key) {
  assert(initialized_);
  auto hash = hashtable_.find(key);
  if (hash == hashtable_.end())
    throw logic_error("a key was not found in Storage::put_block(const size_t&)");
  auto p = hash->second;
  int64_t size = p.second - p.first + 1;
  DataType one = 1.0;
  NGA_Acc64(ga_, &p.first, &p.second, dat.get(), &size, &one);
  // MPI_Acc
}


template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block() const {
  return get_block_(generate_hash_key());
}

template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block(const Index& i0) const {
  return get_block_(generate_hash_key(i0));
}

template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block(const Index& i0, const Index& i1) const {
  return get_block_(generate_hash_key(i0, i1));
}

template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block(const Index& i0, const Index& i1, const Index& i2) const {
  return get_block_(generate_hash_key(i0, i1, i2));
}

template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block(const Index& i0, const Index& i1, const Index& i2, const Index& i3) const {
  return get_block_(generate_hash_key(i0, i1, i2, i3));
}

template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block(const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                          const Index& i4) const {
  return get_block_(generate_hash_key(i0, i1, i2, i3, i4));
}

template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block(const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                          const Index& i4, const Index& i5) const {
  return get_block_(generate_hash_key(i0, i1, i2, i3, i4, i5));
}

template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block(const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                          const Index& i4, const Index& i5, const Index& i6) const {
  return get_block_(generate_hash_key(i0, i1, i2, i3, i4, i5, i6));
}

template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block(const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                          const Index& i4, const Index& i5, const Index& i6, const Index& i7) const {
  return get_block_(generate_hash_key(i0, i1, i2, i3, i4, i5, i6, i7));
}

template<typename DataType>
unique_ptr<DataType[]> StorageIncore<DataType>::get_block(vector<Index> i) const {
  return get_block_(generate_hash_key(i));
}


template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat) {
  put_block_(dat, generate_hash_key());
}

template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat, const Index& i0) {
  put_block_(dat, generate_hash_key(i0));
}

template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1) {
  put_block_(dat, generate_hash_key(i0, i1));
}

template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2) {
  put_block_(dat, generate_hash_key(i0, i1, i2));
}

template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3) {
  put_block_(dat, generate_hash_key(i0, i1, i2, i3));
}

template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                                           const Index& i4) {
  put_block_(dat, generate_hash_key(i0, i1, i2, i3, i4));
}

template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                                           const Index& i4, const Index& i5) {
  put_block_(dat, generate_hash_key(i0, i1, i2, i3, i4, i5));
}

template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                                           const Index& i4, const Index& i5, const Index& i6) {
  put_block_(dat, generate_hash_key(i0, i1, i2, i3, i4, i5, i6));
}

template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                                           const Index& i4, const Index& i5, const Index& i6, const Index& i7) {
  put_block_(dat, generate_hash_key(i0, i1, i2, i3, i4, i5, i6, i7));
}

template<typename DataType>
void StorageIncore<DataType>::put_block(const unique_ptr<DataType[]>& dat, vector<Index> i) {
  put_block_(dat, generate_hash_key(i));
}


template<typename DataType>
void StorageIncore<DataType>::add_block(const unique_ptr<DataType[]>& dat) {
  add_block_(dat, generate_hash_key());
}

template<typename DataType>
void StorageIncore<DataType>::add_block(const unique_ptr<DataType[]>& dat, const Index& i0) {
  add_block_(dat, generate_hash_key(i0));
}

template<typename DataType>
void StorageIncore<DataType>::add_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1) {
  add_block_(dat, generate_hash_key(i0, i1));
}

template<typename DataType>
void StorageIncore<DataType>::add_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2) {
  add_block_(dat, generate_hash_key(i0, i1, i2));
}

template<typename DataType>
void StorageIncore<DataType>::add_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3) {
  add_block_(dat, generate_hash_key(i0, i1, i2, i3));
}

template<typename DataType>
void StorageIncore<DataType>::add_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                                           const Index& i4) {
  add_block_(dat, generate_hash_key(i0, i1, i2, i3, i4));
}

template<typename DataType>
void StorageIncore<DataType>::add_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                                           const Index& i4, const Index& i5) {
  add_block_(dat, generate_hash_key(i0, i1, i2, i3, i4, i5));
}

template<typename DataType>
void StorageIncore<DataType>::add_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                                           const Index& i4, const Index& i5, const Index& i6) {
  add_block_(dat, generate_hash_key(i0, i1, i2, i3, i4, i5, i6));
}

template<typename DataType>
void StorageIncore<DataType>::add_block(const unique_ptr<DataType[]>& dat, const Index& i0, const Index& i1, const Index& i2, const Index& i3,
                                                                           const Index& i4, const Index& i5, const Index& i6, const Index& i7) {
  add_block_(dat, generate_hash_key(i0, i1, i2, i3, i4, i5, i6, i7));
}


template<typename DataType>
void StorageIncore<DataType>::zero() {
  assert(initialized_);
  GA_Zero(ga_);
}


template<typename DataType>
void StorageIncore<DataType>::scale(const DataType& a) {
  assert(initialized_);
  GA_Scale(ga_, const_cast<DataType*>(&a));
}


template<typename DataType>
StorageIncore<DataType>& StorageIncore<DataType>::operator=(const StorageIncore<DataType>& o) {
  if (!initialized_ && o.initialized_)
    initialize();
  GA_Copy(o.ga_, ga_);
  return *this;
}


template<typename DataType>
bool StorageIncore<DataType>::is_local(const size_t key) const {
  auto iter = hashtable_.find(key);
  assert(iter != hashtable_.end());
  int64_t lo = iter->second.first;
  int64_t nodeid = NGA_Nodeid();
  // MPI_Comm_split_type
  return (nodeid+1 < blocks_.size()) && (lo >= blocks_[nodeid] && lo < blocks_[nodeid+1]);
}


template<typename DataType>
void StorageIncore<DataType>::ax_plus_y(const DataType& a, const StorageIncore<DataType>& o) {
  assert(initialized_);
  DataType one = 1.0;
  GA_Add(const_cast<DataType*>(&a), o.ga_, &one, ga_, ga_);
}


template<>
double StorageIncore<double>::dot_product(const StorageIncore<double>& o) const {
  assert(initialized_);
  return GA_Ddot(ga_, o.ga_);
}


template<>
complex<double> StorageIncore<complex<double>>::dot_product(const StorageIncore<complex<double>>& o) const {
  assert(initialized_);
  // GA_Zdot is zdotu, so we have to compute this manually
  complex<double> sum = 0.0;
  for (auto& i : hashtable_) {
    int64_t lo = i.second.first;
    int64_t hi = i.second.second;
    if (is_local(i.first)) {
      unique_ptr<complex<double>[]> a = get_block_(i.first);
      unique_ptr<complex<double>[]> b = o.get_block_(i.first);
      sum += blas::dot_product(a.get(), hi-lo+1, b.get());
    }
  }
  mpi__->allreduce(&sum, 1);
  return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// explict instantiation at the end of the file
template class StorageIncore<double>;
template class StorageIncore<complex<double>>;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
