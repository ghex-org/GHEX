#ifndef INCLUDED_GHEX_OBJ_WRAP_HPP
#define INCLUDED_GHEX_OBJ_WRAP_HPP

#include <cstdint>
#include <memory>
#include <typeinfo>
#include <iostream>

namespace gridtools {
    namespace ghex {
        namespace bindings {

            class obj_wrapper {
	
            public:

                /** interface that queries the stored object type */
                struct obj_type_info {
                    virtual ~obj_type_info() = default;
                    virtual std::type_info const &type_info() const noexcept = 0;
                };

                /** actual object storage */
                template <class T>
                struct obj_storage : obj_type_info{
                    T m_obj;
                    obj_storage(T const &obj) : m_obj(obj) {}
                    obj_storage(T &&obj) : m_obj(std::move(obj)) {}
                    std::type_info const &type_info() const noexcept override { return typeid(T); }
                };
	
                std::unique_ptr<obj_type_info> m_obj_storage;

                obj_wrapper(obj_wrapper &&) = default;

                template <class Arg, class Decayed = typename std::decay<Arg>::type>
                obj_wrapper(Arg &&arg) : m_obj_storage(new obj_storage<Decayed>(std::forward<Arg>(arg))) {}

                std::type_info const &type_info() const noexcept { return m_obj_storage->type_info(); }

                template <class T>
                friend T* get_object_ptr(obj_wrapper *src);
            };

            /** get the object without performing type checks:
             *	assume that has already been done and the cast is legal */
            template <class T>
            T* get_object_ptr_unsafe(obj_wrapper *src) {
                return &reinterpret_cast<obj_wrapper::obj_storage<T> *>(src->m_obj_storage.get())->m_obj;
            }
    
        }
    }
}
    
#endif  /* INCLUDED_GHEX_OBJ_WRAP_HPP */
