////////////////////////////////////////////////////////////////////////
// "supermacro" code for doing some registration stuff for Serializers.
//
// See s11n::io::register_serializer<>() for a function which does the
// same thing as this supermacro (it's easier to use and doesn't
// add as much back-end overhead).
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////
// NOTE: this file does not use a conventional BLAH_H_INCLUDED guard.
// Yes, that's on purpose.
////////////////////////////////////////////////////////////////////////
// The s11n header files are expected to have been included by the
// time this file is ever included.
//
// usage:
// define:
// SERIALIZER_TYPE serializer_class
// SERIALIZER_BASE base_of_serializer (or SERIALIZER_TYPE)
// SERIALIZER_MAGIC_COOKIE optional "#!/magic/cookie/string"
// SERIALIZER_ALIAS optional "alias_string" for classloading
//
// #include <this_file>
//
// After each include all of these macros are unset so that they may
// be immediately re-used for another registration.
//
// The cookie is required if s11n is to be able to load
// your files transparently. An alias is convenient for s11nconvert
// and such, but not required.
////////////////////////////////////////////////////////////////////////

#include <stdlib.h> // abort()

#define DEBUG_REG 0
#if DEBUG_REG
#  include <s11n.net/s11n/s11n_debuggering_macros.hpp> // CERR macro
#endif


#ifndef SERIALIZER_TYPE
#    error "SERIALIZER_TYPE is not set. Set it to the type you want to proxy before including this file!"
#endif

#ifndef SERIALIZER_NAME
#    error "SERIALIZER_NAME must be set to the string form of SERIALIZER_TYPE"
#endif

#ifndef SERIALIZER_BASE
#    error "SERIALIZER_BASE must be the base-most type of SERIALIZER_TYPE (may be the same)."
#endif

#include <s11n.net/s11n/classload.hpp>

namespace { // anonymous namespace is important for complex linking reasons.



#ifndef s11n_SERIALIZER_REG_CONTEXT_DEFINED
#define s11n_SERIALIZER_REG_CONTEXT_DEFINED 1
        ///////////////////////////////////////////////////////////////
        // we must not include this more than once per compilation unit
        ///////////////////////////////////////////////////////////////
        // A unique (per SERIALIZER_BASE/per compilation unit) space to assign
        // a bogus value for classloader registration purposes (see
        // the classloader docs for a full description of how this
        // works).
        template <typename Context>
        struct serializer_reg_context
        {
                typedef Context context;
                static bool placeholder;
                static void reg()
                {
                        CERR << "ACHTUNG: serializer_reg_context<context> >()"
                             << " is not specialized, which means that registration hasn't been done.\n"
                             << "For instructions see: " << __FILE__ << "\n";
                        abort();
                }
        };
        template <typename Context> bool serializer_reg_context<Context>::placeholder = false;
#endif
// !s11n_SERIALIZER_REG_CONTEXT_DEFINED
////////////////////////////////////////////////////////////////////////////////

        template <>
        struct serializer_reg_context< SERIALIZER_TYPE >
        {
                typedef SERIALIZER_TYPE context;
                static bool placeholder;
                static void reg()
                {
                        std::string cname = SERIALIZER_NAME;
#if DEBUG_REG
                        CERR << "\nRegistering Serializer: " << cname << "\n"
#  ifdef SERIALIZER_MAGIC_COOKIE
                             << "cookie="<< SERIALIZER_MAGIC_COOKIE << "\n"
#  endif
#  ifdef SERIALIZER_ALIAS
                             << "alias="<< SERIALIZER_ALIAS << "\n"
#  endif
                                ; // CERR
#endif // DEBUG_REG

#ifdef SERIALIZER_ABSTRACT
                        ::s11n::cl::classloader_register_abstract< SERIALIZER_BASE >( cname );
#  undef SERIALIZER_ABSTRACT
#else
                        ::s11n::cl::classloader_register< SERIALIZER_BASE, SERIALIZER_TYPE >( cname );
#endif
#ifdef SERIALIZER_MAGIC_COOKIE
                        ::s11n::cl::classloader_alias< SERIALIZER_BASE >( SERIALIZER_MAGIC_COOKIE, cname );
#  undef SERIALIZER_MAGIC_COOKIE
#endif
#ifdef SERIALIZER_ALIAS
                        ::s11n::cl::classloader_alias< SERIALIZER_BASE >( SERIALIZER_ALIAS, cname );
#  undef SERIALIZER_ALIAS
#endif
                }
        };

        bool serializer_reg_context< SERIALIZER_TYPE >::placeholder = (
                                                                       serializer_reg_context< SERIALIZER_TYPE >::reg()
                                                                       ,
                                                                       true
                                                                       );

} // anon namespace 

////////////////////////////////////////////////////////////////////////////////
// end proxy code for [SERIALIZER]
////////////////////////////////////////////////////////////////////////////////
#undef SERIALIZER_TYPE
#undef SERIALIZER_NAME
#undef DEBUG_REG
