#if 0 // this file is obsolete, as of version 0.9.14
////////////////////////////////////////////////////////////////////////
// "supermacro" code for registering a Data Node type with
// the built-in Serializers.
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////
// Example usage:
// 
// #define S11N_NODE_TYPE my::node_type
// #define S11N_NODE_TYPE_NAME "my::node_type"
// #include <s11n.net/s11n/reg_node_type.hpp>
//
////////////////////////////////////////////////////////////////////////
// NOTE: this file does not use a conventional BLAH_H_INCLUDED guard.
// Yes, that's on purpose.
////////////////////////////////////////////////////////////////////////


#ifndef S11N_NODE_TYPE
#    error "S11N_NODE_TYPE is not set. Set it to the type you want to proxy before including this file!"
#endif

#ifndef S11N_NODE_TYPE_NAME
#    error "S11N_NODE_TYPE_NAME is not set. Set it to the string form of S11N_NODE_TYPE!"
#endif


#define S11N_DEFAULT_BASE_SERIALIZER s11n::io::serializer< S11N_NODE_TYPE >

#define SERIALIZER_TYPE S11N_DEFAULT_BASE_SERIALIZER
#define SERIALIZER_BASE S11N_DEFAULT_BASE_SERIALIZER
#define SERIALIZER_ABSTRACT 1
#define SERIALIZER_NAME ("s11n::io::serializer<" S11N_NODE_TYPE_NAME ">")
#include <s11n.net/s11n/proxy/reg_serializer.hpp>

#ifndef s11n_REG_NODE_TYPE_FORCE_INSTANTIATION
#define s11n_REG_NODE_TYPE_FORCE_INSTANTIATION 1

namespace s11n { namespace internal {
        /**
           This function exists to force instantiations of all
           "known" Serializers.  We do this because if they
           are registered via client-included header files
           they add a huge amount of time to compiles.
           Registrations for all Serializers for one NodeType
           add about 23 seconds on my 2GHz PC!
           
           Having this here allows us to compile these at library
           compilation time, instead of cutting in to client-side
           compilation times.
        */
        template <typename NodeT>
        void serializer_registration_init()
        {

#define SERINST(SerT,CLNM,ALIAS)                                 \
                ::s11n::io::register_serializer< SerT< NodeT > >( CLNM, ALIAS );
                // Set up the built-in Serializers for NodeT...
//                 SERINST(::s11n::io::funxml_serializer,"s11n::io::funxml_serializer","funxml");
//                 SERINST(::s11n::io::parens_serializer,"s11n::io::parens_serializer","parens");
//                 SERINST(::s11n::io::compact_serializer,"s11n::io::compact_serializer","compact");
//                 SERINST(::s11n::io::funtxt_serializer,"s11n::io::funtxt_serializer","funtxt");
//                 SERINST(::s11n::io::simplexml_serializer,"s11n::io::simplexml_serializer","simplexml");
// #if s11n_HAVE_LIBEXPAT
//                 SERINST(::s11n::io::expat_serializer,"s11n::io::expat_serializer","expat");
// #endif

#undef SERINST
        }

} } // namespace s11n::internal

namespace {
        /**
           Acts as a per-NodeType placeholder, to avoid ODR violations.
        */
        template <typename T>
        struct bogus_serializer_placeholder {
                static int placeholder;
        };
}

#endif // s11n_REG_NODE_TYPE_FORCE_INSTANTIATION

namespace {
        int bogus_serializer_placeholder< S11N_NODE_TYPE >::placeholder = 
                ( s11n::internal::serializer_registration_init< S11N_NODE_TYPE > (), 0 );

}



#undef S11N_NODE_TYPE
#undef S11N_NODE_TYPE_NAME
#undef S11N_DEFAULT_BASE_SERIALIZER

#endif
