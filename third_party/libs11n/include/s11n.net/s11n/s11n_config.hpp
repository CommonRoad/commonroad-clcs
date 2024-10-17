#ifndef s11n_CONFIG_HPP_INCLUDED
#define s11n_CONFIG_HPP_INCLUDED 1
// Template file for s11n_config.hpp - project-wide defines.  Make
// your changes in s11n_config.hpp.at, not in s11n_config.hpp, as the
// configure script will use s11n_config.hpp.at to create
// s11n_config.hpp.

////////////////////////////////////////////////////////////////////////
// Code which wants to check for s11n's inclusion should check
// for:
#define s11n_S11N_INCLUDED 1
// This does not mean that ALL components are loaded, only that
// some part of it has been. Classes may use this to conditionally
// include their s11n registrations.
// As of version 1.0.2, a more configurable approach is to check
// against s11n_S11N_LIBRARY_VERSION_HEX, which contains the
// version number encoded as a hex int. e.g., 1.0.8 == 0x010008
////////////////////////////////////////////////////////////////////////


#define s11n_S11N_PACKAGE_NAME "s11n"
#define s11n_S11N_LIBRARY_VERSION "1.2.10"
#define s11n_S11N_LIBRARY_VERSION_HEX (0x010210)
#define s11n_PACKAGE_RELEASE_CODENAME "The Software Still Known As S11n"
#define s11n_PACKAGE_LICENSE "Public Domain"
#define s11n_PACKAGE_URL "http://s11n.net/"
#define s11n_PACKAGE_EMAIL_ADDRESS "s11n-devel@lists.sourceforge.net"


#if defined(WIN32)
 /* have libexpat XML parser? */
#  define s11n_CONFIG_HAVE_LIBEXPAT 0
 /* have libzfstream? */
#  define s11n_CONFIG_HAVE_ZFSTREAM 0
 /* Enable s11n::plugin module? */
#  define s11n_CONFIG_ENABLE_PLUGINS (1)
#else
#  define s11n_CONFIG_HAVE_LIBEXPAT 0
#  define s11n_CONFIG_HAVE_ZFSTREAM 0
#  define s11n_CONFIG_ENABLE_PLUGINS (1)
#endif

/* Shared paths for s11n and clients. */
#if defined(WIN32)
#  define s11n_CONFIG_SHARED_DIR std::string("C:\\s11n.net\\shared")
#  define s11n_CONFIG_LIB_DIR std::string("C:\\s11n.net\\lib")
#else
#  define s11n_CONFIG_SHARED_DIR std::string("/home/stephan/share/s11n")
#  define s11n_CONFIG_LIB_DIR std::string("/home/stephan/lib/s11n")
#endif

#define s11n_S11NLITE_DEFAULT_SERIALIZER_TYPE_NAME std::string("s11n::io::funtxt_serializer")

#endif // s11n_CONFIG_HPP_INCLUDED
