////////////////////////////////////////////////////////////////////////
// A "supermacro"-style classloader registration snippet for the
// s11n::fac layer.
////////////////////////////////////////////////////////////////////////
// Macros to be set before including this file:
//
// S11N_FACREG_TYPE: type to be registered.
//
// S11N_FACREG_TYPE_NAME: string name to register S11N_FACREG_TYPE with.
//
// S11N_FACREG_INTERFACE_TYPE: base-most type of S11N_FACREG_TYPE.
// Defaults to S11N_FACREG_TYPE.
//
// OPTIONAL/special-case macros:
//
// S11N_FACREG_TYPE_IS_ABSTRACT:
// This macro should be used only when S11N_FACREG_TYPE is an abstract type.
// If S11N_FACREG_TYPE is an abstract base class, define this
// macro to any value. If this is set then a no-op specialization
// of create_hook<> is installed so no calls to (new S11N_FACREG_TYPE) will
// be made by the default factories. 
//
//
// All of the above macros are #undef'd by this file, so it may be
// included multiple times in succession.
////////////////////////////////////////////////////////////////////////

#ifndef S11N_FACREG_TYPE
#  error "You must define S11N_FACREG_TYPE before including this file."
#endif

#ifndef S11N_FACREG_TYPE_NAME
#  error "You must define S11N_FACREG_TYPE_NAME before including this file."
#endif

#ifndef S11N_FACREG_INTERFACE_TYPE
#  define S11N_FACREG_INTERFACE_TYPE S11N_FACREG_TYPE
#endif

#include <s11n.net/s11n/export.hpp>

#ifdef S11N_FACREG_TYPE_IS_ABSTRACT
// Install a specialization of Factory's default factory type,
// so that we won't try to instantiate a S11N_FACREG_TYPE object.
// For the non-abstract case we will rely on the default create_hook
// implementation (or a client-installed specialization).
namespace s11n { namespace fac {
	/**
	   create_hook specialization to install a no-op
	   factory. Intended for use with abastract types.
	*/
	template <>
	struct S11N_EXPORT_API create_hook< S11N_FACREG_INTERFACE_TYPE , S11N_FACREG_TYPE >
	{
		typedef S11N_FACREG_INTERFACE_TYPE * result_type;
		typedef S11N_FACREG_TYPE actual_type;
		/** Returns 0. */
		static result_type create()
		{
			return 0;
		}
	};
} } // namespace s11n::fac
#endif // S11N_FACREG_TYPE_IS_ABSTRACT

namespace {

#  ifndef s11n_FACTORY_REG_CONTEXT_DEFINED
#  define s11n_FACTORY_REG_CONTEXT_DEFINED 1
	///////////////////////////////////////////////////////////////
	// we must not include this more than once per compilation unit
	/**
	   A unique (per Context/per compilation unit) space to assign
	   a bogus value for classloader registration purposes (see
	   the classloader docs for a full description of how this
	   works).
	*/
        template <typename Context1, typename Context2 = Context1>
	struct S11N_EXPORT_API s11n_factory_reg_context
	{
		s11n_factory_reg_context()
		{
		    if( true == placeholder ){} // weird workaround, just to reference the var. 
		}
		/**
		   Placeholder variable for automatic factory
		   registration.
		*/
		static bool placeholder;
	};
#  endif // !s11n_FACTORY_REG_CONTEXT_DEFINED

	/**
	   Placeholder variable for automatic factory registration.
	*/
	template <>
	bool /* do we need S11N_EXPORT_API here??? */
	s11n_factory_reg_context< S11N_FACREG_INTERFACE_TYPE, S11N_FACREG_TYPE >::placeholder= (
								    ::s11n::fac::register_factory< S11N_FACREG_INTERFACE_TYPE >(
															   std::string( S11N_FACREG_TYPE_NAME ),
															   ::s11n::fac::create_hook<
															   S11N_FACREG_INTERFACE_TYPE,
															   S11N_FACREG_TYPE
															   >::create
								    ),
								      true);
	// DAMN, that's fugly!
} // anon namespace

#ifdef S11N_FACREG_TYPE_IS_ABSTRACT
#  undef S11N_FACREG_TYPE_IS_ABSTRACT
#endif
#undef S11N_FACREG_TYPE
#undef S11N_FACREG_INTERFACE_TYPE
#undef S11N_FACREG_TYPE_NAME
