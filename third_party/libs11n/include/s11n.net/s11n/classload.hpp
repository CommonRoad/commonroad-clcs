#ifndef s11n_cl_S11N_CLASSLOAD_HPP_INCLUDED
#define s11n_cl_S11N_CLASSLOAD_HPP_INCLUDED 1

#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // S11N_TRACE()
#include <s11n.net/s11n/factory.hpp>
#include <s11n.net/s11n/export.hpp>


namespace s11n {
        /**
           The s11n::cl namespace encapsulates the public
           classloader-related API for libs11n.

           Note that the registration functions in this namespace
           register with the s11n::fac family of classes. Clients wishing to use
           their own factories should:

           - register with their appropriate classloader.

           - specialize the object_factory class template
           so that it forwards calls to their classloader.

           This layer is used for classloading anything which s11n
           needs to dynamically load, including:

           - Serializables

           - Serializers

           - FlexLexers

           It supports loading types via DLLs if the optional s11n::plugin
	   layer is enabled.

	   Changes from 1.0.x to 1.1.x:

	   - Now internally uses s11n::factory_mgr and s11n::plugin
	   instead of the cllite API.

	   Changed in 1.2.2:

	   - operator() implementation code was moved to classload.tpp.
        */
        namespace cl {

		/**
		   A default object factory functor for use
		   with the s11n::s11n_traits type.
	   
		   Clients may specialize this to return objects from
		   their own factories. By default it uses s11n::fac's
		   mini-framework, and thus can load any type registered with
		   that API.
		*/
	    template <typename InterfaceBase>
		struct S11N_EXPORT_API object_factory
		{
			/** Same as InterfaceBase. */
			typedef InterfaceBase base_type;

			/**
			   The default implementation returns
			   ::s11n::fac::factory<InterfaceBase>().create(key). The
			   caller owns the returned pointer, which may
			   be 0.

			   If the underlying factory does not have the
			   requested class and s11n is built with the
			   s11n::plugin layer, then
			   s11n::plugin::open(classname) is used to
			   search for a DLL. Under the s11n classloader
			   model, opening a plugin will register classes
			   defined in the plugin back with the factory.

			   This function propagates exceptions if the
			   underlying factory's create() feature throws.

			   a) s11n_exceptions are passed on as-is.

			   b) std::exceptions are translated to
			   s11n::factory_exceptions, keeping the
			   what() text.

			   c) Unknown exceptions (...) are translated
			   to s11n::factory_exceptions with an
			   unspecified non-empty what() text.

			*/
			base_type * operator()( const std::string & key ) const;
		};

                /**
		   Returns object_factory<InterfaceBase>(key).
                */
                template <typename InterfaceBase>
                InterfaceBase * classload( const std::string key );

                /**
                   Registers a factory with InterfaceBase's
                   classloader.

		   Trivia: 'register' is a reserved word in C++,
		   inherited from C, so this function has an unduly
		   long name. register() was the first choice.
                */
                template <typename InterfaceBase>
                void classloader_register( const std::string & classname, InterfaceBase * (*factory_func)() );

		/**
		   Registers a default factory which returns a SubType
		   object. SubType must be a public subtype of
		   InterfaceBase, or must be InterfaceBase, and must
		   not be abstract.
		*/
                template <typename InterfaceBase, typename SubType>
                inline void classloader_register_subtype( const std::string & classname )
		{
			classloader_register<InterfaceBase>( classname, 
							     ::s11n::fac::create_hook<InterfaceBase,SubType>::create
							     );
		}

                /**
                   Registers InterfaceBase with its own classloader
                   using a default factory. InterfaceBase must be creatable
		   with 'new', or the default factory implementation must
		   be specialized to accomodate the abstract class.
                */
                template <typename InterfaceBase>
                inline void classloader_register_base( const std::string & classname )
                {
                        classloader_register_subtype<InterfaceBase,InterfaceBase>( classname );
                }


		namespace Detail
		{
			/** Internal no-op factory. Always returns 0. */
			template <typename T>
			inline T * noop_factory() { return 0; }
		}

                /**
                   Registers InterfaceBase as an abstract type. That is, it
                   assigns it a factory which always returns 0.

                   If types you register with classloader_register()
                   (and friends) have an abstract InterfaceBase then that
                   InterfaceBase should be registered via this function (or
                   equivalent).

		   Note that this is also suitable for registering
		   abstract subtypes of an abstract type, but
		   InterfaceBase MUST be the type passed here (not the
		   subtype's type), or the registrations will go to
		   the subtype's classloader, which normally isn't
		   what you want to do.
                */
                template <typename InterfaceBase>
                inline void classloader_register_abstract( const std::string & classname )
                {
 			classloader_register<InterfaceBase>( classname, Detail::noop_factory<InterfaceBase> );
                }


		/**
		   Aliases the given classname with the underlying
		   factory layer, such that classload<InterfaceBase>(_alias)
		   will return the same as classload<InterfaceBase>(classname).

		   A tip for remembering the order of the arguments:
		   it follows the same order as when assigning an alias
		   via a Unix shell: alias thealias="the original string"

		   Added in 1.1.0.

		   Maintenance reminder: this function relies directly
		   on the s11n::fac API, which isn't "really"
		   public. This functionality is commonly useful when
		   hand-registering types, thus we have a public-API
		   wrapper around this fac-layer-specific feature.
		*/
                template <typename InterfaceBase>
                inline void classloader_alias( const std::string & _alias,
					       const std::string & classname)
                {
			::s11n::fac::factory<InterfaceBase>().aliases().alias( _alias, classname );
                }


        } // namespace cl
} // namespace s11n

#include <s11n.net/s11n/classload.tpp> // implementations
#endif // s11n_cl_S11N_CLASSLOAD_HPP_INCLUDED
