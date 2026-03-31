#pragma once
//file full of either forward declarations 
//or types used in caf cuda


#include <caf/intrusive_ptr.hpp>
#include <variant>
#include <vector>
#include <stdexcept>

#if defined(_MSC_VER)
  #define CAF_CUDA_EXPORT __declspec(dllexport)
#else
  #define CAF_CUDA_EXPORT __attribute__((visibility("default")))
#endif


//memory access flags, required for identifying which
//gpu buffers are input and output buffers
#define IN 0 
#define IN_OUT 1
#define OUT 2
#define NOT_IN_USE -1



namespace caf::cuda {

// Forward declarations and intrusive_ptr aliases

class device;
using device_ptr = caf::intrusive_ptr<device>;

class platform;
using platform_ptr = caf::intrusive_ptr<platform>;

class program;
using program_ptr = caf::intrusive_ptr<program>;

template <class T>
class mem_ref;

template <class T>
using mem_ptr = caf::intrusive_ptr<mem_ref<T>>;

// Forward declare manager, command, and actor_facade

class CAF_CUDA_EXPORT manager;

template <class Actor, class... Ts>
class command;

template <bool PassConfig, class... Ts>
class actor_facade;

} // namespace caf::cuda


// === buffer_variant and output_buffer outside namespace or inside as needed ===

using buffer_variant = std::variant<std::vector<char>, std::vector<int>, std::vector<float>, std::vector<double>>;

struct output_buffer {
  buffer_variant data;
};


// === Wrapper types for in/out/in_out with default ctor, union safely used ===

//represents a readonly buffer on the gpu
template <typename T>
class in_impl {
private:
  T scalar_;
  std::vector<T> owned_buf_; // non-empty only when constructed from rvalue vector
  const T* ptr_;
  size_t size_;
  bool is_scalar_;
  bool moved_from_ = false;

  void check_valid() const {
    if (moved_from_)
      throw std::runtime_error("Use-after-move detected in in_impl");
  }

  // Re-seat ptr_ to this object's own storage after copy/move construction
  // or assignment.  For non-owning views (const-ref ctor path), copies the
  // external pointer from the source.
  void reseat_ptr(const in_impl& src) noexcept {
    if (is_scalar_)
      ptr_ = &scalar_;
    else if (!owned_buf_.empty())
      ptr_ = owned_buf_.data();
    else
      ptr_ = src.ptr_; // non-owning view: copy the external pointer
  }

public:
  using value_type = T;

  // scalar ctors
  in_impl()
    : scalar_{}, owned_buf_{}, ptr_{&scalar_}, size_{1}, is_scalar_{true} {}

  explicit in_impl(const T& val)
    : scalar_{val}, owned_buf_{}, ptr_{&scalar_}, size_{1}, is_scalar_{true} {}

  // zero-copy vector view — caller must ensure the vector outlives this object
  explicit in_impl(const std::vector<T>& buf)
    : scalar_{}, owned_buf_{}, ptr_{buf.data()}, size_{buf.size()}, is_scalar_{false} {}

  // owning rvalue overload — takes ownership of the vector, no dangling pointer
  explicit in_impl(std::vector<T>&& buf)
    : scalar_{}, owned_buf_{std::move(buf)}, ptr_{owned_buf_.data()},
      size_{owned_buf_.size()}, is_scalar_{false} {}

  in_impl(const in_impl& other)
    : scalar_{other.scalar_},
      owned_buf_{other.owned_buf_},
      size_{other.size_},
      is_scalar_{other.is_scalar_},
      moved_from_{other.moved_from_} {
    reseat_ptr(other);
  }

  in_impl& operator=(const in_impl& other) {
    if (this != &other) {
      scalar_     = other.scalar_;
      owned_buf_  = other.owned_buf_;
      size_       = other.size_;
      is_scalar_  = other.is_scalar_;
      moved_from_ = other.moved_from_;
      reseat_ptr(other);
    }
    return *this;
  }

  in_impl(in_impl&& other) noexcept
    : scalar_{std::move(other.scalar_)},
      owned_buf_{std::move(other.owned_buf_)},
      size_{other.size_},
      is_scalar_{other.is_scalar_},
      moved_from_{other.moved_from_} {
    reseat_ptr(other);
    other.moved_from_ = true;
  }

  in_impl& operator=(in_impl&& other) noexcept {
    if (this != &other) {
      scalar_     = std::move(other.scalar_);
      owned_buf_  = std::move(other.owned_buf_);
      size_       = other.size_;
      is_scalar_  = other.is_scalar_;
      moved_from_ = other.moved_from_;
      reseat_ptr(other);
      other.moved_from_ = true;
    }
    return *this;
  }

  bool is_scalar() const {
    check_valid();
    return is_scalar_;
  }

  const T& getscalar() const {
    check_valid();
    if (!is_scalar_)
      throw std::runtime_error("in_impl does not hold scalar");
    return scalar_;
  }

  const T* data() const {
    check_valid();
    return ptr_;
  }

  size_t size() const {
    check_valid();
    return size_;
  }

  // Required for CAF serialization compatibility
  std::vector<T> get_buffer() const {
    check_valid();
    return std::vector<T>(ptr_, ptr_ + size_);
  }
};

//represents a write only buffer on the gpu
template <typename T>
class out_impl {
private:
  T scalar_;
  int size_ = 1;
  bool is_scalar_ = true;
  bool moved_from_ = false;

  void check_valid() const {
    if (moved_from_)
      throw std::runtime_error("Use-after-move detected in out_impl");
  }

public:
  using value_type = T;

  out_impl() : scalar_{}, size_(1), is_scalar_(true) {}

  // allocate GPU buffer of given size
  explicit out_impl(int size)
    : scalar_{}, size_(size), is_scalar_(false) {}

  explicit out_impl(const std::vector<T>& buf)
    : scalar_{}, size_(static_cast<int>(buf.size())), is_scalar_(false) {}

  out_impl(const out_impl&) = default;
  out_impl& operator=(const out_impl&) = default;

  out_impl(out_impl&& other) noexcept {
    scalar_ = other.scalar_;
    size_ = other.size_;
    is_scalar_ = other.is_scalar_;
    other.moved_from_ = true;
  }

  out_impl& operator=(out_impl&& other) noexcept {
    if (this != &other) {
      scalar_ = other.scalar_;
      size_ = other.size_;
      is_scalar_ = other.is_scalar_;
      other.moved_from_ = true;
    }
    return *this;
  }

  bool is_scalar() const {
    check_valid();
    return is_scalar_;
  }

  const T* data() const {
    check_valid();
    return &scalar_;
  }

  size_t size() const {
    check_valid();
    return static_cast<size_t>(size_);
  }

  // compatibility with CAF serialization
  std::vector<T> get_buffer() const {
    check_valid();
    return std::vector<T>(size_);
  }
};


template <typename T>
class in_out_impl {
private:
  T scalar_;
  std::vector<T> owned_buf_; // non-empty only when constructed from rvalue vector
  const T* ptr_;
  size_t size_;
  bool is_scalar_;
  bool moved_from_ = false;

  void check_valid() const {
    if (moved_from_)
      throw std::runtime_error("Use-after-move detected in in_out_impl");
  }

  void reseat_ptr(const in_out_impl& src) noexcept {
    if (is_scalar_)
      ptr_ = &scalar_;
    else if (!owned_buf_.empty())
      ptr_ = owned_buf_.data();
    else
      ptr_ = src.ptr_;
  }

public:
  using value_type = T;

  in_out_impl()
    : scalar_{}, owned_buf_{}, ptr_{&scalar_}, size_{1}, is_scalar_{true} {}

  explicit in_out_impl(const T& val)
    : scalar_{val}, owned_buf_{}, ptr_{&scalar_}, size_{1}, is_scalar_{true} {}

  explicit in_out_impl(const std::vector<T>& buf)
    : scalar_{}, owned_buf_{}, ptr_{buf.data()}, size_{buf.size()}, is_scalar_{false} {}

  explicit in_out_impl(std::vector<T>&& buf)
    : scalar_{}, owned_buf_{std::move(buf)}, ptr_{owned_buf_.data()},
      size_{owned_buf_.size()}, is_scalar_{false} {}

  in_out_impl(const in_out_impl& other)
    : scalar_{other.scalar_},
      owned_buf_{other.owned_buf_},
      size_{other.size_},
      is_scalar_{other.is_scalar_},
      moved_from_{other.moved_from_} {
    reseat_ptr(other);
  }

  in_out_impl& operator=(const in_out_impl& other) {
    if (this != &other) {
      scalar_     = other.scalar_;
      owned_buf_  = other.owned_buf_;
      size_       = other.size_;
      is_scalar_  = other.is_scalar_;
      moved_from_ = other.moved_from_;
      reseat_ptr(other);
    }
    return *this;
  }

  in_out_impl(in_out_impl&& other) noexcept
    : scalar_{std::move(other.scalar_)},
      owned_buf_{std::move(other.owned_buf_)},
      size_{other.size_},
      is_scalar_{other.is_scalar_},
      moved_from_{other.moved_from_} {
    reseat_ptr(other);
    other.moved_from_ = true;
  }

  in_out_impl& operator=(in_out_impl&& other) noexcept {
    if (this != &other) {
      scalar_     = std::move(other.scalar_);
      owned_buf_  = std::move(other.owned_buf_);
      size_       = other.size_;
      is_scalar_  = other.is_scalar_;
      moved_from_ = other.moved_from_;
      reseat_ptr(other);
      other.moved_from_ = true;
    }
    return *this;
  }

  bool is_scalar() const {
    check_valid();
    return is_scalar_;
  }

  const T& getscalar() const {
    check_valid();
    if (!is_scalar_)
      throw std::runtime_error("in_out_impl does not hold scalar");
    return scalar_;
  }

  const T* data() const {
    check_valid();
    return ptr_;
  }

  std::size_t size() const {
    check_valid();
    return size_;
  }

  // required for CAF serialization
  std::vector<T> get_buffer() const {
    check_valid();
    return std::vector<T>(ptr_, ptr_ + size_);
  }
};

// === Aliases ===

//readonly buffer
template <typename T>
using in = in_impl<T>;

//writeonly buffer
template <typename T>
using out = out_impl<T>;


//readwrite buffer
template <typename T>
using in_out = in_out_impl<T>;

// Helper to get raw type inside wrapper
template <typename T>
struct raw_type {
  using type = T;
};

template <typename T>
struct raw_type<in<T>> {
  using type = T;
};

template <typename T>
struct raw_type<out<T>> {
  using type = T;
};

template <typename T>
struct raw_type<in_out<T>> {
  using type = T;
};

template <typename T>
struct raw_type<caf::cuda::mem_ptr<T>> {
  using type = T;
};

template <typename T>
using raw_t = typename raw_type<T>::type;


