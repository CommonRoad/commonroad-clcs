#ifndef s11n_plugin_CONFIG_HPP_INCLUDED
#define s11n_plugin_CONFIG_HPP_INCLUDED 1

// Configuration options specific to the s11n::plugin layer. These
// options are not in the main s11n_config.hpp due to the circular
// dependencies that would bring about as i add plugins support to the
// core.

#if defined(WIN32)
#  define s11n_CONFIG_PLUGINS_PATH std::string("C:\\s11n.net\\lib\\plugins")
#  define s11n_CONFIG_HAVE_LIBLTDL 0
#  define s11n_CONFIG_HAVE_LIBDL 0
#  define s11n_CONFIG_DLL_EXTENSION std::string(".dll")
#else
#  define s11n_CONFIG_PLUGINS_PATH std::string(".:/home/stephan/share/s11n/plugins:/home/stephan/lib/s11n")
#  define s11n_CONFIG_HAVE_LIBLTDL 0
#  define s11n_CONFIG_HAVE_LIBDL 0
#  define s11n_CONFIG_DLL_EXTENSION std::string(".so:.dynlib")
#endif

#endif // s11n_plugin_CONFIG_HPP_INCLUDED
