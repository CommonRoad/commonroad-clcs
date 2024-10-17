////////////////////////////////////////////////////////////////////////
// A supermacro to generate some partial template specializations for
// s11n-proxying class templates taking two parameterized types. Works
// with std::map and std::pair-like types, with the appropriate
// proxies. See the various proxy files, like std/map.hpp and
// std/pair.hpp, for how to use it.
//
// Achtung: only suitable for monomorphic or base-most template types.
////////////////////////////////////////////////////////////////////////

#ifndef S11N_TEMPLATE_TYPE
#error "You must define S11N_TEMPLATE_TYPE before including this file. e.g., to std::map or std::multimap."
#endif

#ifndef S11N_TEMPLATE_TYPE_NAME
#error "You must define S11N_TEMPLATE_TYPE_NAME before including this file. e.g., to \"map\" or \"multimap\"."
#endif

#ifndef S11N_TEMPLATE_TYPE_PROXY
#  error "You must define S11N_TEMPLATE_TYPE_PROXY before including this file. e.g., ::s11n::map::map_serializable_proxy"
#endif

#ifndef S11N_TEMPLATE_TYPE_DESER_PROXY
#  define S11N_TEMPLATE_TYPE_DESER_PROXY S11N_TEMPLATE_TYPE_PROXY
#endif

////////////////////////////////////////////////////////////////////////
// The S11N_TEMPLATE_TYPENAME_Tn macros are used when one or more of
// the template parameters are concrete types, as opposed to typenames.
// Define any of S11N_TEMPLATE_TYPENAME_Tn to a type name, e.g.:
// #  define S11N_TEMPLATE_TYPENAME_T1 unsigned int
////////////////////////////////////////////////////////////////////////

#ifndef S11N_TEMPLATE_TYPENAME_T1
#  define S11N_TEMPLATE_TYPENAME_T1 typename
#endif

#ifndef S11N_TEMPLATE_TYPENAME_T2
#  define S11N_TEMPLATE_TYPENAME_T2 typename
#endif


#define S11N_TEMPLATE_TYPE_Q S11N_TEMPLATE_TYPE< T1, T2 >

#ifndef S11N_BASE_TYPE
#define S11N_BASE_TYPE S11N_TEMPLATE_TYPE_Q
#endif

#define S11N_TYPENAMES S11N_TEMPLATE_TYPENAME_T1 T1, \
	S11N_TEMPLATE_TYPENAME_T2 T2
namespace s11n {

    /**
       s11n_traits<> specialization for template types taking two
       template parameters, like std::map and std::pair
       types. (Yes, std::map can take more, but it is not commonly
       used that way.)
    */
    template < S11N_TYPENAMES >
    struct S11N_EXPORT_API s11n_traits < S11N_TEMPLATE_TYPE_Q, S11N_BASE_TYPE >
    {
	typedef S11N_TEMPLATE_TYPE_Q serializable_type;
	typedef S11N_BASE_TYPE serializable_interface_type;
	typedef S11N_TEMPLATE_TYPE_PROXY serialize_functor;
	typedef S11N_TEMPLATE_TYPE_DESER_PROXY deserialize_functor;
	typedef ::s11n::default_cleanup_functor< S11N_TEMPLATE_TYPE_Q > cleanup_functor;
	typedef ::s11n::cl::object_factory<serializable_interface_type> factory_type;
	static bool cl_reg_placeholder; 
	static const std::string class_name( const serializable_type * )
	{
	    if( cl_reg_placeholder == true ){} // just to reference it. w/o this cl reg never happens :(
	    return S11N_TEMPLATE_TYPE_NAME;
	}
    };
    template < S11N_TYPENAMES >
    bool s11n_traits<
	S11N_TEMPLATE_TYPE_Q, S11N_BASE_TYPE
	>::cl_reg_placeholder = (
				 ::s11n::cl::classloader_register_base< S11N_TEMPLATE_TYPE_Q >(S11N_TEMPLATE_TYPE_NAME),
				 true
				 );
    
    
} // namespace s11n


#undef S11N_TEMPLATE_TYPE_Q
#undef S11N_TEMPLATE_TYPE_PROXY
#undef S11N_TEMPLATE_TYPE_DESER_PROXY
#undef S11N_TEMPLATE_TYPE_NAME
#undef S11N_TEMPLATE_TYPE
#undef S11N_TEMPLATE_TYPENAME_T1
#undef S11N_TEMPLATE_TYPENAME_T2
#undef S11N_BASE_TYPE
#undef S11N_TYPENAMES
