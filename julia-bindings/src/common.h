#pragma once

#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

class Object {
public:
    virtual ~Object() = default;
    virtual const char* name() const = 0;

    template<typename T>
    const T& unwrap() const {
        if (const T* p = dynamic_cast<const T*>(this)) {
            return *p;
        }

        std::stringstream s;
        s << "invalid object (expecting: " << typeid(T).name() << ", gotten: " << this->name()
          << ")";
        throw std::runtime_error(s.str());
    }
};

template<typename T>
class ObjectImpl: public Object, public T {
public:
    template<typename... Args>
    ObjectImpl(Args&&... args) : T(std::forward<Args>(args)...) {}

    const char* name() const {
        return typeid(T).name();
    }
};

template<typename T>
ObjectImpl<std::decay_t<T>>* new_object(T&& arg) {
return new ObjectImpl<std::decay_t<T>>(std::forward<T>(arg));
}

template<typename T, typename... Args>
ObjectImpl<T>* new_object(Args&&... args) {
    return new ObjectImpl<T>(std::forward<Args>(args)...);
}

extern "C" void __attribute__ ((noreturn)) jl_errorf(const char *fmt, ...);

template<typename F>
auto catch_exceptions(F fun) -> decltype(fun()) {
    try {
        return fun();
    } catch (const std::exception& msg) {
        // Not sure how to pass the error to julia. Abort for now.
        jl_errorf("COMPAS error occurred: %s\n", msg.what());
    } catch (...) {
        jl_errorf("COMPAS error occurred: %s\n", "unknown exception");
    }

    fprintf(stderr, "Fatal error: unreachable path, `jl_errorf` should not return");
    std::abort();
}