////////////////////////////////////////////////////////////////////////
// reg_s11n_traits.hpp
//
// Supermacro for registering s11n proxies for the s11n API, using
// the s11n_traits<> technique introduced in 0.9.3.
//
// C macros used/expected by this file:
//
// S11N_TYPE: type to be registered
//
// S11N_TYPE_NAME: stringified name of S11N_TYPE
//
// S11N_BASE_TYPE: if S11N_TYPE subclasses a regisitered S11N_TYPE,
// pass it here. If this is set then the rest of the following
// variables need not be set, as they are instead inherited from
// the S11N_BASE_TYPE's registration info.
//
// S11N_SERIALIZE_FUNCTOR: functor type responsible for handling calls
// to serialize<S11N_TYPE>(). Default is
// s11n::default_serialize_functor, which simply calls
// S11N_TYPE::operator()( NodeT & ).
//
// S11N_DESERIALIZE_FUNCTOR: functor type responsible for handling calls
// to serialize(). Defaults to S11N_SERIALIZE_FUNCTOR. In practice
// this never needs to be explicitely set.
//
// S11N_FACTORY_TYPE: functor which is responsible for instantiating
// objects of type S11N_TYPE. Default behaviour is to use
// s11n::cl::object_factory< S11N_BASE_TYPE >, which is fine for the
// majority of cases.
//
// THINGS TO KNOW:
//
// - If the default factory type is NOT used, the caller is
// responsible for registering S11N_TYPE with the appropriate factory,
// using the key S11N_TYPE_NAME.
//
////////////////////////////////////////////////////////////////////////
// The s11n header files are expected to have been included by the
// time this file is ever included.
//
//
// Sample usage:
//
// #define S11N_TYPE MyBaseType
// #define S11N_TYPE_NAME "MyBaseType"
// #include [this file]
//
// repeat for each type. For subtypes of MyBaseType, do:
//
// #define S11N_TYPE MySubType
// #define S11N_TYPE_NAME "MySubType"
// #define S11N_BASE_TYPE MyBaseType
// #include [this file]
//
// Always use MyBaseType as the S11N_BASE_TYPE, no matter
// now far-inherented MySubType is from MyBaseType.
//
// If S11N_BASE_TYPE is not set then the following macros
// are also used, otherwise they are ignored, inherited from
// the base implementation:
//
// #define S11N_SERIALIZE_FUNCTOR some_serializer_functor
// #define S11N_DESERIALIZE_FUNCTOR some_deserializer_functor
// #define S11N_CLEANUP_FUNCTOR some_cleanup_functor
//
// If S11N_DESERIALIZE_FUNCTOR is defined, that is used as the
// Deserialization proxy, otherwise S11N_SERIALIZE_FUNCTOR functor is
// used for both cases.// If neither are set then
// ::s11n::default_serialize_functor is installed, and S11N_TYPE must
// conform to that proxy's expectations.
//
// The cleanup functor must conform to s11n_traits::cleanup_functor
// conventions. The default is to use
// s11n::default_cleanup_functor<S11N_BASE_TYPE>.
//
// After each include all of these macros are unset so that they may
// be immediately re-used for another registration.
////////////////////////////////////////////////////////////////////////


#ifndef S11N_TYPE
#  error "S11N_TYPE must be set before including this file."
#endif

#ifndef S11N_TYPE_NAME
#  ifdef S11N_NAME // older convention
#    define S11N_TYPE_NAME S11N_NAME
#    undef S11N_NAME
#  else
#    error "S11N_TYPE_NAME must be set before including this file. Set it to the stringified form of S11N_TYPE."
#  endif
#endif


// S11N_BASE_TYPE only needs to be set when registering subtypes of an
// already-registered base type...
#ifdef S11N_BASE_TYPE
#  define S11N_INHERIT 1
#else
#  define S11N_BASE_TYPE S11N_TYPE
#  define S11N_INHERIT 0
#  ifndef S11N_SERIALIZE_FUNCTOR
#    define S11N_SERIALIZE_FUNCTOR ::s11n::default_serialize_functor
#  endif
#  ifndef S11N_DESERIALIZE_FUNCTOR
#    define S11N_DESERIALIZE_FUNCTOR S11N_SERIALIZE_FUNCTOR
#  endif
#  ifndef S11N_CLEANUP_FUNCTOR
#    define S11N_CLEANUP_FUNCTOR ::s11n::default_cleanup_functor< S11N_BASE_TYPE >
#  endif
#endif // S11N_BASE_TYPE


#ifndef S11N_FACTORY_TYPE
     // then register a factory of our own...
#    define S11N_REG_DEFAULTFAC 1
#    define S11N_FACTORY_TYPE ::s11n::cl::object_factory< S11N_BASE_TYPE >
#else
     // assume the user knows what he's doing...
#    define S11N_REG_DEFAULTFAC 0
#endif // S11N_FACTORY_TYPE

////////////////////////////////////////////////////////////////////////
// Set up s11n_traits<> specialization...
namespace s11n {

        template <>
        struct s11n_traits < S11N_TYPE, S11N_BASE_TYPE >
#if S11N_INHERIT
                : public s11n_traits < S11N_BASE_TYPE >
#endif
        {
#if ! S11N_INHERIT
                typedef S11N_TYPE serializable_type;
                typedef S11N_SERIALIZE_FUNCTOR serialize_functor;
                typedef S11N_DESERIALIZE_FUNCTOR deserialize_functor;
                typedef S11N_FACTORY_TYPE factory_type;
                typedef S11N_CLEANUP_FUNCTOR cleanup_functor;
#endif // ! S11N_INHERIT
                typedef S11N_BASE_TYPE serializable_interface_type;
                static const std::string class_name( const serializable_type * /* instance_hint */ )
		{
			return S11N_TYPE_NAME;
		}

        };

} // namespace s11n
// end s11n_traits<>
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
// We don't actually need to install a factory registration for abstract
// types, but we do because there is a very subtle difference between 
// registering and not registering one:
// the factory layer might do a DLL lookup for our abstract type if it
// doesn't have a factory, but won't do so if it finds a factory but
// the factory returns 0.
#if S11N_REG_DEFAULTFAC
#  define S11N_FACREG_TYPE S11N_TYPE
#  define S11N_FACREG_TYPE_NAME S11N_TYPE_NAME
#  define S11N_FACREG_INTERFACE_TYPE S11N_BASE_TYPE
#    ifdef S11N_ABSTRACT_BASE
#        define S11N_FACREG_TYPE_IS_ABSTRACT 1
#    endif
#  include <s11n.net/s11n/factory_reg.hpp>
#endif // S11N_REG_DEFAULTFAC
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
// clean up for the next #include ...
#undef S11N_TYPE
#undef S11N_TYPE_NAME
#undef S11N_BASE_TYPE
#undef S11N_SERIALIZE_FUNCTOR
#undef S11N_DESERIALIZE_FUNCTOR
#undef S11N_FACTORY_TYPE
#undef S11N_INHERIT
#undef S11N_REG_DEFAULTFAC
#undef S11N_CLEANUP_FUNCTOR
#ifdef S11N_ABSTRACT_BASE
#  undef S11N_ABSTRACT_BASE
#endif
