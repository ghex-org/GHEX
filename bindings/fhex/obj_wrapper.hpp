#ifndef INCLUDED_GHEX_FORTRAN_OBJ_WRAP_HPP
#define INCLUDED_GHEX_FORTRAN_OBJ_WRAP_HPP

#include <cstdint>
#include <memory>
#include <typeinfo>
#include <iostream>

namespace gridtools {
    namespace ghex {
        namespace fhex {

            class obj_wrapper {

            public:

                /** base class for stored object type */
                struct obj_storage_base {
                    virtual ~obj_storage_base() = default;
                };

                /** actual object storage */
                template <class T>
                struct obj_storage : obj_storage_base{
                    T m_obj;
                    obj_storage(T const &obj) : m_obj(obj) {}
                    obj_storage(T &&obj) : m_obj(std::move(obj)) {}
                };

                std::unique_ptr<obj_storage_base> m_obj_storage;

                obj_wrapper(obj_wrapper &&) = default;

                template <class Arg, class Decayed = typename std::decay<Arg>::type>
                obj_wrapper(Arg &&arg) : m_obj_storage(new obj_storage<Decayed>(std::forward<Arg>(arg))) {}
            };

            /** get the object without performing type checks:
             *	assume that has already been done in Fortran and the cast is legal */
            template <class T>
            T* get_object_ptr_unsafe(obj_wrapper *src) {
                return &reinterpret_cast<obj_wrapper::obj_storage<T> *>(src->m_obj_storage.get())->m_obj;
            }

        }
    }
}

#endif  /* INCLUDED_GHEX_FORTRAN_OBJ_WRAP_HPP */
