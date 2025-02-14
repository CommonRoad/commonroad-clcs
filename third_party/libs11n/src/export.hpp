#ifndef s11n_net_S11N_EXPORT_HPP_INCLUDED
#define s11n_net_S11N_EXPORT_HPP_INCLUDED
/**********************************************************************
This file defines some macros for "exporting" symbols when the code is
compiled as a shared library (DLL). Windows platforms, and possibly
others, need to be told exactly what symbols are to be exported.

S11N_EXPORT_API should be used when declaring a class which needs to be
visible from outside the DLL:

  class S11N_EXPORT_API my_class { ... };

Free functions:

  S11N_EXPORT_API int my_func() { ... }


**********************************************************************/
#ifdef WIN32
// #warning "Exporting Windows-style!"
 #  ifdef S11N_EXPORTS
#    define S11N_EXPORT_API __declspec(dllexport)
 #  else
 #    define S11N_EXPORT_API __declspec(dllimport)
 #  endif
#else
#    define S11N_EXPORT_API
#endif

//#endif // s11n_net_S11N_EXPORT_HPP_INCLUDED

/*
#ifdef S11NEXPORT
# undef S11NEXPORTAPI
# ifndef S11NEXPORTAPI
#  if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#    if defined(STATIC_LINKED)
#      define S11NEXPORTAPI
#    else
#      define S11NEXPORTAPI __declspec(dllexport)
#    endif
#  else
#    if defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#      define S11NEXPORTAPI __attribute__ ((visibility("default")))
#    else
#      define S11NEXPORTAPI
#    endif
#  endif
# endif
#else 
# ifndef S11NEXPORTAPI
#  if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#    if defined(STATIC_LINKED)
#      define S11NEXPORTAPI
#    else
#      define S11NEXPORTAPI __declspec(dllimport)
#    endif
#  else
#    if defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#      define S11NEXPORTAPI __attribute__ ((visibility("default")))
#    else
#      define S11NEXPORTAPI
#    endif
#  endif
# endif
#endif
*/
#endif

