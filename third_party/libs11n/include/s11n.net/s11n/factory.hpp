#ifndef s11n_net_s11n_FACTORY_H_INCLUDED
#define s11n_net_s11n_FACTORY_H_INCLUDED 1
// Author: stephan beal <stephan@s11n.net>
// Various parts between 2003 and 2005.
// License: Public Domain

#include <string>
#include <map>

#include <functional>

#include <s11n.net/s11n/export.hpp>
#include <s11n.net/s11n/phoenix.hpp> // i don't like this dep, but i also don't like
	                       // what happens in some cases when i don't use
        	               // phoenix. :/

namespace s11n {

	namespace sharing {
		/**
		   Internal marker type.
		*/
		struct factory_context {};
	}

	/**
	   The s11n::fac namespace contains s11n's "private" object
	   factory manager.  This code should not be considered to be
	   part of the public APIs: use the s11n::cl API for your
	   factory-related needs unless absolutely
	   necessary. Unfortunately, classloading can be intrusive
	   enough that this API might sometimes need to be leaked into
	   client- or non-core library code. The example which
	   immediately comes to mind is clients which want to get a
	   list of all registered factories: this can only be done by
	   querying the factory itself for them.

	   The factory manager layer acts as a registration point for
	   factories for near-arbitrary classes. Classes compatible
	   with this layer must currently meet the following requirements:

	   - DefaultConstructable.

	   - Must register with this layer. In s11n, this is done as
	   part of the normal s11n registration process. Clients
	   needing to manually register types should use the s11n::cl
	   class registering/loading API.

	   - Must be legal: delete instanceOfObject; This is really a
	   usage convention, and not a hard/fast rule of this type.
	*/
	namespace fac {

		/**
		   create_hook is a helper object factory for the factory_mgr
		   API. It is used by that type to create instances of specific types.
		   It may be specialized to provide non-default object construction.

		   General conventions:

		   SubT must derive from (or be) T and must be Default
		   Constructuable on the heap. In short, the following must be
		   able to succeed:
	   
		   <pre>
		   T * foo = new SubT;
		   </pre>

		   Clients may freely specialize this type to hook their
		   factories in, and the above 'new' requirement need
		   not be imposed by client-side specializations. For example,
		   specializations are used to provide no-op factories for
		   abstract types, where 'new T' cannot work.
		*/

		template < class T, class SubT >
		struct S11N_EXPORT_API create_hook
		{
			/**
			   The type returned by create() and
			   operator().
			*/
			typedef T * result_type;

			/**
			   A typedef for the second template parameter for this
			   type.
			*/
			typedef SubT actual_type;
			
			/**
			   This creates a new SubT, which is assumed to be a
			   subclass of T.  It can be used as a factory for
			   factory & class_loader.
			   
			   If T or SubT are abstract types, you must
			   specialize this type such that create() returns 0
			   for those.  That is, we "simulate" creation of
			   abstract types by returning 0.

			   The caller owns the returned pointer, which
			   may be 0.
			*/
			static result_type create()
			{
				return new actual_type;
			}

			/**
			   Same as create().
			*/
			result_type operator()() const
			{
				return this->create();
			}
		};


		/**
		   A helper class to alias one token to another.

		   AliasedType requires:

		   - Must be suitable for use as both key and value
		   types for a std::map, with all that that implies
		   (e.g., operator<() must be well-behaved).

		*/
		template <typename AliasedType>
		class S11N_EXPORT_API aliaser
		{
		public:
			typedef AliasedType key_type;
			typedef AliasedType value_type;
			/**
			   A map type for storing lookup key aliases.
			*/
			typedef std::map<key_type,value_type> alias_map_type;

			/**
			   Aliases 'alias' as an equivalent of 'isthesameas'.
			*/
			void alias( const key_type & alias, const value_type & isthesameas )
			{
				this->aliases()[alias] = isthesameas;
			}

			/**
			   Returns the map of aliases.
			*/
			alias_map_type & aliases() { return this->m_map; }

			/** Const overload. */
			const alias_map_type & aliases() const { return this->m_map; }

			typedef typename alias_map_type::iterator iterator;
			typedef typename alias_map_type::const_iterator const_iterator;

			/** Begin iterator for aliases(). */
			iterator begin() { return this->m_map.begin(); }
			/** Begin const_iterator for aliases(). */
			const_iterator begin() const { return this->m_map.begin(); }

			/** End iterator for aliases(). */
			iterator end() { return this->m_map.end(); }
			/** End const_iterator for aliases(). */
			const_iterator end() const { return this->m_map.end(); }

			/**
			   Expands the given alias recursively. If a circular
			   alias is detected, the last expansion is returned
			   (i.e., same as _alias). If no expansion is found
			   then _alias is returned.
			*/
			value_type expand( const key_type & _alias ) const
			{
				typename alias_map_type::const_iterator cit = this->m_map.find( _alias ),
					cet = this->m_map.end();
				if( cet == cit )
				{
					return _alias;
				}
				value_type exp = (*cit).second;
				while( 1 )
				{
					cit = this->m_map.find( exp );
					if( cet == cit )
					{
						return exp;
					}
					exp = (*cit).second;
					if( exp == _alias )
					{ // circular aliasing
						return exp;
					}
				}
				return exp;
			}
		private:
			alias_map_type m_map;

		};

		/**
		   The factory_mgr class is essentially a static classloader,
		   capable of loading classes by using registered factories
		   for a given set of keys (e.g., class names).

		   Classloaders, at least in my experience, need to be able to
		   load all classes which derive from some given type. Without
		   a common base class, one can't safely attempt to cast from
		   an arbitrary pointer to the type we want to load. That's
		   where the InterfaceT parameter comes in. All objects
		   instantiated via this loader must inherit (be-a) from
		   InterfaceT, or must literally be InterfaceT.

		   KeyType is a type which specifies the type of key used to
		   look up classes, defaulting to std::string.

		   For this implementation, both InterfaceT and
		   KeyType must be Default Constructable, and
		   InterfaceT must be constructable on the heap (e.g.,
		   via new InterfaceT()).

		   InterfaceT must be a "plain type", without any
		   pointer, reference or const qualifications.

		   Sample usage:

		   <pre>
		   factory_mgr<MyInterface> & fac = factory_mgr<MyInterface>::instance();
		   fac.register_factory( "my_key", s11n::fac::create_hook<MyInterface,MyClass>::create );
		   MyInterface *foo = fac.create( "some_key" ); // == NULL
		   foo = fac.create( "my_key" ); // == a new MyClass object
		   </pre>

		   Note that all instances of factory_mgr share the
		   same factories and aliases maps. This is a design
		   decision which is intended to simplify usage of the
		   type and ensure consistency of state across module
		   boundaries. In effect, the default implementation
		   is a Monostate type, which each instance sharing
		   the same data.

		   Trivia: above i mentioned "casting", but this
		   implementation requires no casting of any type,
		   neither in the library nor in client code (unless
		   the client explicitely needs to do so for their own
		   purposes).
		*/
		template < class InterfaceT,
			   class KeyType = std::string >
		class S11N_EXPORT_API factory_mgr
		{
		private:
			/**
			   Convenience typedef.
			*/
			typedef factory_mgr< InterfaceT, KeyType > ThisType;
		public:

			/**
			   A typedef for the KeyType used by this class.
			*/
			typedef KeyType key_type;

			/**
			   A typedef for the InterfaceT used by this class.
			   For conformance with the Adaptable Unary Functor
			   model
			*/
			typedef InterfaceT value_type;
			/**
			   Same as (InterfaceT *).
			*/
			typedef InterfaceT * result_type;


			factory_mgr() {}

			virtual ~factory_mgr() throw() {}

			/**
			   A type used to map classname aliases for
			   this factory manager.
			*/
			typedef aliaser<key_type> aliaser_type;
			/**
			   The type of factories used by this class: a
			   function taking void and returning (value_type
			   *). See factory_map().

			   todo: implement proper functor support.
			*/
			typedef result_type ( *factory_function_type ) ();

			/**
			   Internal container type used for mapping keys to
			   factories.
			*/
			typedef std::map < key_type, factory_function_type > factory_map_type;


			/**
			   Returns the container of classname
			   aliases. In this implementation the aliases
			   are shared amongst all instances of
			   this type, because that aligns it with the
			   shared factory map.
			*/
			aliaser_type & aliases()
			{
				return ::s11n::Detail::phoenix<aliaser_type, ThisType>::instance();
			}

			/**
			   Const overload.
			*/
			const aliaser_type & aliases() const
			{
				return ::s11n::Detail::phoenix<aliaser_type, ThisType>::instance();
			}


			/**
			   Tries to instantiate an instance of value_type
			   using the given key. Returns 0 if no object
			   could be loaded for the given key.

			   Subtypes are free to implement, e.g., DLL lookups.

			   This implementation calls aliases().expand(_key) to
			   expand any aliased class
			   names. Subclasses/specializations "should" do the
			   same, but are not strictly required to.
			*/
			virtual result_type create( const key_type & _key )
			{
				key_type key = this->aliases().expand( _key );
				typename factory_map_type::const_iterator it = factory_map().find( key );
				if ( it != factory_map().end() )	// found a factory?
				{
					return ( it->second ) ();	// run our factory.
				}
				return 0;
			}


			/**
			   Returns create(key).
			*/
			result_type operator()( const key_type & key )
			{
				return this->create( key );
			}

			/**
			   Simply calls delete obj. Subclasses are
			   free to do custom accounting, garbage
			   collection, or whatever, by overriding
			   this.

			   Note that it is NOT practical to expect all clients
			   to call this in order to destroy their objects, so
			   the utility of this function is HIGHLY arguable.

			   Also be aware that the simple delete
			   behaviour is not suitable if InterfaceT is
			   a populated container of dumb pointers, or
			   otherwise contains unowned pointers.

			   Subclasses are free to "suggest" a
			   must-destroy() policy if they wish, but
			   cannot expect anyone to actually follow
			   that wish. In pedantic terms, however, one
			   should destroy an object from the same code
			   module and thread which created it, to
			   avoid low-level problems involving crossing
			   module/thread boundaries. That's what
			   Sutter and Alexandrescu say, anyway, and
			   i'm not one to argue with them.
			*/
			virtual void destroy( result_type obj )
			{
				delete obj;
			}


			/**
			   Registers a factory using the given
			   key. Note that fp may not be declared as
			   returning a type other than (value_type *),
			   but the actual object it creates may be a
			   polymorphic subclass of value_type. See the
			   create_hook class for a factory which does
			   this subtype-to-base conversion.

			   For ABSTRACT interface types, you "should"
			   register a no-op factory which simply
			   returns 0.  The difference between
			   registering a no-op factory and registering
			   NO factory is that when create() is called,
			   if NO factory is found, this type is free
			   to implement extra logic like DLL lookups
			   to try to find the type (which is futile
			   for an abstract type). However, if it has a
			   registered factory, it will not do this (it
			   doesn't care if the factory returns 0 or
			   not).

			   Subclasses are free to add behaviour, like
			   instrumentation, but should not forget to
			   actually carry through on the registration
			   part (e.g., by calling this
			   implementation).
			*/
			virtual void register_factory( const key_type & key, factory_function_type fp )
			{
				// CERR << "register_factory("<<key<<",facfunc)\n";
				// ^^^ i keep that around for when i'm trying to figure out whether a DLL
				// is actually registering itself.
				factory_map().insert( typename factory_map_type::value_type( key, fp ) );
			}

			/**
			   Returns the internal key-to-factory map. It is safe
			   for clients to modify this except in multi-threaded
			   environments, and then all guarantees go out the
			   window. That said, it should never be necessary for
			   clients to use this.

			   The default implementation returns the same map for
			   all instances of ThisType.  The reasoning behind
			   using static factory maps is essentially this: we
			   can only have one definition of each type. We
			   normally want factories to always return an
			   instance constructed in the same way. If we allow
			   multiple factories per type we might introduce
			   hard-to-track inconsistencies in client code, where
			   different factories than intended are accidentally
			   used. OTOH, private factory maps would open up some
			   interesting possibilities.

			   TODO (well, to consider): use a hook or
			   Traits type to allow the user to replace
			   this object with his own, so that he can
			   control the scope of the map more
			   fully. Overkill. YAGNI. Christian
			   Prochnow's experience in the P::Classes
			   tree suggests that that would be Bad,
			   anyway, because clients linking from
			   different sources can end up with different
			   instances of the map. This has never been
			   witnessed using phoenix<> to share
			   instances, however.
			*/
			factory_map_type & factory_map()
			{
				return ::s11n::Detail::phoenix<factory_map_type, ThisType>::instance();
			}

			/** Const overload. */
			const factory_map_type & factory_map() const
			{
				return ::s11n::Detail::phoenix<factory_map_type, ThisType>::instance();
			}

			/**
			   Returns true if the given key is registered
			   (alias expansions are considered). This is
			   sometimes useful for checking whether a
			   factory needs to be re-registered, which is
			   sometimes necessary post-main(), when the
			   internal map gets hosed before clients are
			   done using it.

			   The constness is arguable:
			   subclasses/specializations could arguably
			   do pre-lookups for DLLs (or similar) here,
			   and might need non-const behaviour. As a
			   consolation, we have made create()
			   non-const, so (provides()==false) can
			   symantically mean "we need to do a DLL
			   lookup".
			*/
			virtual bool provides( const key_type & key ) const
			{
				return factory_map().end() != factory_map().find( this->aliases().expand(key) );
			}

			/**
			   Returns a shared reference to a factory.

			   The s11n core always calls this function to get factory
			   instances.
			*/
			static factory_mgr & instance()
			{
 				// return ::s11n::fac::instance_hook<ThisType>::instance();
				// ^^^ Christian Prochnow reports that this approach can cause
				// multiple objects to be created across DLLs, which is something
				// we definately don't want to happen!
				return ::s11n::Detail::phoenix<ThisType>::instance();
			}

		}; // class factory_mgr


		/**
		   Returns factory_mgr<InterfaceT, std::string>::instance().

		   All s11n-internal factory_mgr operations operate on
		   one of these objects.
		*/
		template <typename InterfaceT>
		inline factory_mgr<InterfaceT, std::string> &
		factory()
		{
			return factory_mgr<InterfaceT,
				std::string
				>::instance();
		}

		/**
		   Registers classname with InterfaceT using the given factory.
		*/
		template <typename InterfaceT>
		inline void register_factory( const std::string & classname, InterfaceT *(*factory_function)() )
		{
			factory<InterfaceT>().register_factory( classname, factory_function );
		}

		/**
		   Registers classname with InterfaceT using a default
		   factory: create_hook<InterfaceT,ImplT>::create().

		   If ImplT is abstract then this function will fail
		   unless create_hook is specialized to not create an
		   object.
		*/
		template <typename InterfaceT>
		inline void register_factory( const std::string & classname )
		{
			factory<InterfaceT>().register_factory( classname, create_hook<InterfaceT,InterfaceT>::create );
		}

 		/**
 		   Registers a factory creating ImplT objects with the
 		   InterfaceT classloader using a default factory.
 		   ImplT may not be abstract unless
 		   create_hook<InterfaceT,ImplT> is specialized such
 		   that it's create() function simply returns 0 (or
 		   otherwise does not call: new ImplT).
 		*/
 		template <typename InterfaceT, typename ImplT>
 		inline void register_subtype( const std::string & classname )
 		{
 			factory<InterfaceT>().register_factory( classname, create_hook<InterfaceT,ImplT>::create );
 		}

		namespace Detail {
			/**
			   Always returns 0. Intended for use as a
			   factory for abstract types.
			*/
			template <typename T>
			inline T * null_factory()
			{
				return 0;
			}
		}

		/**
		   Registers a no-op factory for the given name and
		   InterfaceT. The factory will always return 0.
		*/
		template <typename InterfaceT>
		void register_abstract( const std::string & classname )
		{
			factory<InterfaceT>().register_factory( classname, Detail::null_factory<InterfaceT> );
			// ^^^ compiler bug? If i fully qualify ::s11n::Detail::null_factory<...>
			// i get a parse error
		}

	
		/**
		   Returns the same as factory<InterfaceT>().create( classname ).
		*/
		template <typename InterfaceT>
		inline InterfaceT * create( const std::string & classname )
		{
			return factory<InterfaceT>().create( classname );
		}

	} // namespace fac

} // namespace s11n

#endif // s11n_net_s11n_FACTORY_H_INCLUDED
