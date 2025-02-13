#ifndef s11n_DEBUGGERING_MACROS_H
#define s11n_DEBUGGERING_MACROS_H 1

// CERR is a drop-in replacement for std::cerr, but slightly more
// decorative.
#ifndef CERR
#define CERR std::cerr << __FILE__ << ":" << std::dec << __LINE__ << " : "
#endif

#ifndef COUT
#define COUT std::cout << __FILE__ << ":" << std::dec << __LINE__ << " : "
#endif

#include <iostream>

//Added by Damien to make Windows compile work
#include <s11n.net/s11n/export.hpp> // S11N_EXPORT_API


////////////////////////////////////////////////////////////////////////
// Debuggering/tracing macros for the s11n internals...
// The xxx_PROFILE_xxx macros are NOT part of the API:
// they are to allow me to quickly switch between various
// debuggering modes.
#define S11N_TRACE_PROFILE_QUIET (::s11n::debug::TRACE_NEVER)
#define S11N_TRACE_PROFILE_DEFAULT (::s11n::debug::TRACE_ERROR | ::s11n::debug::TRACE_WARNING )
#define S11N_TRACE_PROFILE_MAINTAINER (S11N_TRACE_PROFILE_DEFAULT | ::s11n::debug::TRACE_FACTORY_REG )

////////////////////////////////////////////////////////////////////////
// S11N_TRACE_LEVELS defines the default, compiled-in tracing level
// When set to 0 (TRACE_NONE), tracing will be unavailable even if
// trace_mask() is later used to change it, and a smart compiler will
// optimize out all such S11N_TRACE calls.
#ifndef S11N_TRACE_LEVELS // allow client code to change compile-time default
//#  define S11N_TRACE_LEVELS (S11N_TRACE_PROFILE_MAINTAINER)
#  define S11N_TRACE_LEVELS (S11N_TRACE_PROFILE_DEFAULT)
#endif

// The S11N_TRACE macro is described in the s11n::debug namespace docs
#define S11N_TRACE(LVL) if((S11N_TRACE_LEVELS) && ((LVL) & ::s11n::debug::trace_mask())) \
		::s11n::debug::trace_stream() << "S11N_TRACE["<<# LVL<<"]: "<<__FILE__<<":"<<std::dec<<__LINE__<<":\n\t"


namespace s11n {
	/**
	   The s11n::debug namespace holds some code for debugging and tracing
	   s11n internals. It is not intended for client-side use.

	   Debuggering macros:

	   S11N_TRACE_LEVELS is a bitmask of TraceFlags values. It defines
	   which types of tracing are enabled by default. Code which should be
	   "traced" should use the S11N_TRACE macro like this:

	   S11N_TRACE(TRACE_LEVEL) << "output ...\n";

	   The output will only be generated when S11N_TRACE_LEVELS is
	   non-zero and the given TRACE_LEVEL mask matches the current
	   value of trace_mask().

	   The mask may be changed at runtime by using the
	   trace_mask() function, and set the default at compile time
	   by defining S11N_TRACE_LEVELS before including
	   s11n_debuggering_macros.hpp.
	*/
	namespace debug {

		/**
		   For use with the S11N_TRACE macro.
		*/
		enum TraceFlags {
		TRACE_NEVER = 0x00000000, // always off
		TRACE_TRIVIAL = 0x00000001, // absolutely trivial info which mainly serves to clutter the console
		TRACE_INFO = 0x00000002, // flag for 'info' traces
		TRACE_WARNING = 0x00000004, // ditto for 'warning'
		TRACE_ERROR = 0x00000008, // ditto for 'error'
		TRACE_CTOR = 0x00000010, // tracer for ctors
		TRACE_DTOR = 0x00000020, // tracer for dtors
		TRACE_CLEANUP = 0x00000040, // tracer for cleanup-on-failed-deser
		TRACE_FACTORY_REG = 0x00000100, // factory registrations
		TRACE_FACTORY_LOOKUP = 0x00000200, // factory lookups
		TRACE_FACTORY_PLUGINS = 0x00000400, // trace plugin-related stuff
		TRACE_FACTORY = 0x00000F00, // trace all factory ops
		TRACE_IO = 0x00001000, // for s11n::io
		TRACE_NYI =   0x00010000, // NYI == Not Yet Implemented
		TRACE_FIXME = 0x00020000, // FIXME/TODO notices
		TRACE_SATAN = 0x00040000, // for chasing down really nasty buggers
		TRACE_ALWAYS = 0xffffffff // matches all flags except TRACE_NEVER
		};

		/**
		   Sets the active trace mask and returns the previous
		   mask.
		*/
		unsigned long trace_mask( unsigned long f );

		/**
		   Returns the current trace mask.
		*/
		//Added by Damien to make Windows compile work
		S11N_EXPORT_API unsigned long trace_mask();

		/**
		   Returns the ostream used for tracing
		   messages. Default is std::cerr.
		*/
      //Added by Damien to make Windows compile work
		S11N_EXPORT_API std::ostream & trace_stream();

		/** Sets the ostream used for tracing messages. */
		void trace_stream( std::ostream & );

		/**
		   A helper type to temporarily change the debug mask,
		   then revert it at destruction.
		*/
		struct trace_mask_changer
		{
			/**
			   Stores the current trace mask.
			 */
			trace_mask_changer();
			/**
			   Stores the current trace mask then
			   sets then calls trace_mask(m).
			 */
			trace_mask_changer( unsigned long m );
			/**
			   Sets the trace_mask() to the value it
			   had when this object was constructed.
			*/
			~trace_mask_changer();
		private:
			unsigned long m_mask;
		};

	} // namespace
} // namespaces

#endif //  s11n_DEBUGGERING_MACROS_H
