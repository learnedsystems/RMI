#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/*
 * Layout #1: bare array of keys (e.g. for SOSD)
 *   - number of elements
 *   - array
 **/
template<typename K>
class KeyArray {

private:

  size_t size_;
  K* begin;

  // for closing
  int fd;
  void* addr;
  size_t file_size;

public:

  KeyArray(const size_t s, const K* begin) : size_(s), begin(begin), fd(-1), addr(NULL), file_size(0) {}

  KeyArray(const char* filename) {
    // open file
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
      std::cerr << "Error opening " << filename << std::endl;
      exit(1);
    }

    // get file size for mapping
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      std::cerr << "Error obtaining fstat" << std::endl;
      exit(1);
    }
    size_t file_size = sb.st_size;

    // mmap
    void* addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
      std::cerr << "Error mmap data" << std::endl;
      exit(1);
    }
    K* whole_data = reinterpret_cast<K*>(addr);
    
    // interpret data
    size_t data_size = static_cast<size_t>(whole_data[0]);
    K* data = whole_data + 1;

    // assign to class
    this->size_ = data_size;
    this->begin = data;
    this->fd = fd;
    this->addr = addr;
    this->file_size = file_size;
  }

  // constructor when size_ is known and not included in the file
  KeyArray(const char* filename, const size_t data_size) {
    // open file
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
      std::cerr << "Error opening " << filename << std::endl;
      exit(1);
    }

    // get file size for mapping
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      std::cerr << "Error obtaining fstat" << std::endl;
      exit(1);
    }
    size_t file_size = sb.st_size;

    // mmap
    void* addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
      std::cerr << "Error mmap data" << std::endl;
      exit(1);
    }
    K* whole_data = reinterpret_cast<K*>(addr);

    // assign to class
    this->size_ = data_size;
    this->begin = whole_data;
    this->fd = fd;
    this->addr = addr;
    this->file_size = file_size;
  }

  ~KeyArray() {
    if (this->addr != NULL) {
      munmap(this->addr, this->file_size);
    }
    if (this->fd != -1) {
      close(this->fd);
    }
  }

  size_t size() const {
    return this->size_;
  }

  // find rank of this key
  size_t rank(const K &key) const {
    return this->rank_within(key, 0, this->size());
  }

  // find rank of this key within given position bound
  size_t rank_within(const K &key, const size_t left, const size_t right) const {
    // assert(left < right);
    K* left_ptr = this->begin + left;
    K* right_ptr = this->begin + std::min(right, this->size());
    K* true_ptr = std::lower_bound(left_ptr, right_ptr, key);
    if (true_ptr < right_ptr && *true_ptr == key) {
      // found, return rank
      return std::distance(this->begin, true_ptr);
    } else {
      // not found, return size() [equivalent to std::end]
      return this->size();
    }
  }

  K* raw_pointer() {
    return this->begin;
  }

  // get key
  K& operator[](const size_t rank) {
    // assert(rank < this->size()); // segfault
    return this->begin[rank];
  }
  // pointer arithmetic
  K* operator+(const size_t y) {
    // assert(y < this->size()); // segfault
    return this->begin + y;
  }
};

/*
 * Layout #2: key-pointer + value array (e.g. KV storage)
 *   - number of elements
 *   - array of key-pointer
 *   - array of values
 **/

template<typename K, typename V>
struct KeyPointer {
  K key;
  V* pointer;
};


template<typename K, typename V>
class KeyPointerArray {

private:

  KeyPointer<K, V>* begin;
  KeyPointer<K, V>* end;

  struct Comparator {

    const K to_key(const KeyPointer<K, V> &v1) const {
      return v1.key;
    }

    const K to_key(const K &v1) const {
      return v1;
    }

    template<typename T1, typename T2>
    bool operator()(const T1 &v1, const T2 &v2) const {
      return to_key(v1) < to_key(v2);
    }
  };

public:

  KeyPointerArray(KeyPointer<K, V>* begin, KeyPointer<K, V>* end) {
    this->begin = begin;
    this->end = end;
  }

  V* find(K key) const {
    KeyPointer<K, V>* true_ptr = std::lower_bound(this->begin, this->end, key, Comparator());
    if (true_ptr < this->end && true_ptr->key == key) {
      // item found within boundary
      return true_ptr->pointer;
    } else {
      // item not found
      return NULL;
    }
  }
};


template<typename K, typename V>
class MmapKeyValueArray {

private:

public:

};
