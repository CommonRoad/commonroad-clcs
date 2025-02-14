#ifndef s11n_SERIALIZE_HPP_INCLUDED
#define s11n_SERIALIZE_HPP_INCLUDED
////////////////////////////////////////////////////////////////////////
// serialize.hpp:
//
// Defines the core de/serialize() functions (and close friends).
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////

#include <stdexcept>


////////////////////////////////////////////////////////////////////////////////
// NO DEPS ON s11n_node.hpp ALLOWED!
////////////////////////////////////////////////////////////////////////////////

namespace s11n {


	namespace Detail {
		/***
		    s11n_api_marshaler is the internal API marshaller
		    for s11n.  See the lib manual for full
		    details. Client code is not expected to use or
		    specialize this class, but theoretically some
		    cases may call for doing so. In s11n versions
		    prior to 1.0.x, specializing this type was
		    sometimes useful for handling client-side
		    template types, but this is no longer necessary
		    nor encouraged.

		    In the default implementation, s11n_traits<NodeT,SerializableT>
		    is used to marshal the de/serialize() calls.

		    Changed in 1.1.3:

		    - NodeType template param was moved from the static
		    functions to the class.

		    - Moved class from anonymous namespace into s11n::Detail.
		*/
		template <typename NodeType,typename SerializableT>
		struct s11n_api_marshaler
		{
			/**
			   Same as SerializableT.
			*/
			typedef SerializableT serializable_type;

			typedef NodeType node_type;

			/**
			   Returns s11n_traits<serializable_type>::serialize_functor()( dest, src ).

			   This implementation sets dest's class name to
			   s11n_traits<SerializableT>::class_name(&src), which
			   is only guaranteed to work properly for monomorphic
			   types and base-most types of Serialization
			   hierarchies (i.e., the registered
			   bases). Polymorphic Serializable subtypes should
			   set this class name themselves, or via their
			   s11n_traits::class_name() specialization, both
			   described in the library manual.

			*/
			static bool serialize( node_type &dest, const serializable_type & src );

			/**
			   Returns s11n_traits<SerializableT>::deserialize_functor()(src,dest).
			*/
			static bool deserialize( const node_type & src, serializable_type & dest );
		};

		/**
		   A specialization to handle pointer types the same as
		   reference/value types. It simply translates the pointers
		   into references.
		*/
		template <typename NodeType,typename SerializableT>
		struct s11n_api_marshaler<NodeType,SerializableT *>
		{
			/**
			   The SerializableT templatized type, minus any
			   pointer part.
			*/
			typedef SerializableT serializable_type;

			typedef NodeType node_type;

			/**
			   Convenience typedef: this class' quasi-parent type.
			*/
			typedef s11n_api_marshaler<node_type,serializable_type> parent_type;

			/**
			   Returns parent_type::serialize( dest, *src );

			   src must be a valid pointer, else false is returned.
			*/
			static bool serialize( node_type &dest, const serializable_type * const & src );

			/**
			   Returns parent_type::deserialize( src, *dest );

			   dest must be a valid pointer, else false is returned.

			   Reminder to self: if we dupe the code from
			   deserialize(const N&,ST *&), we could
			   potentially provide support for passing a
			   reference to a null pointer here. No need,
			   though, since the public s11n API already
			   provides that.
			*/
			static bool deserialize( const node_type & src, serializable_type * & dest );
		};
	} // namespace Detail


        /**
           Serializes src to target using the default API marshaling
           mechanism.

	   On success it always returns true, else false.

	   If a the underlying operation throws, it will pass on the
	   exception.
        */
        template <typename DataNodeType, typename SerializableT>
        bool serialize( DataNodeType & target, const SerializableT & src );

	/**
	   Calls s11n_traits<SerializableType>::cleanup_functor()(s).

	   This function is declared as no-throw because of its
	   logical role in the destruction process, and dtors are
	   normally prohibited from throwing. Any exceptions caught by
	   this function are silently ignored (a warning might go out
	   to a debug channel, probably cerr, but don't rely on it).

	   SerializableType requirements:

	   - Must be a Serializable. Specifically, it must have an
	   s11n_traits specialization installed.

	   - s11n_traits<SerializableType>::cleanup_functor must be
	   known to work properly for SerializableType. This is core
	   to the whole cleanup functionality, which is core to
	   protecting against leaks in the face of errors.

	   Technically, if the type can be delete()d without leaking
	   pointers, it's safe for use with this function, but this
	   function SHOULD NOT be used as general cleanup tool. It is
	   ONLY intended to be used with REGISTERED Serializables.

	   This function guarantees not to leak when "cleaning up"
	   containers holding unmanaged pointers as long as the
	   associated cleanup_functors do their part. The model is
	   such that once a cleanup_functor is in place for a given
	   type, this function will inherently walk it and invoke the
	   cleanup rules, which includes freeing any pointers along
	   the way.

	   Added in 1.1.3.
	*/
	template <typename SerializableType>
	void cleanup_serializable( SerializableType & s ) throw();

	/**
	   This overload provides cleanup handling for pointer
	   types. This simplifies many algorithms over using
	   s11n_traits<SerializableType>::cleanup_functor directly, as
	   the algorithms do not need to care if they're using
	   pointer-qualified types or not in order to clean them up
	   properly.

	   SerializableType requirements are as for the non-pointered
	   variant of this function, plus:

	   - delete aSerializableTypeInstance; must be well-formed and
	   must neither throw nor invoke undefined behaviour.  (Did
	   you realize that "neither" is an exception to English's
	   "i-before-e" rule?)

	   This function does nothing if s is null, otherwise it calls
	   cleanup_serializable(*s), deletes s, then assigns it to 0.

	   Postcondition: (0 == s)

	   Added in 1.1.3.
	*/
	template <typename SerializableType>
	void cleanup_serializable( SerializableType * & s ) throw();


        /**
	   Intended for use with for_each(), this type cleans up
	   Serializables using cleanup_serializable().

	   Usage:

	   std::for_each( container.begin(), container.end(), cleaner_upper() );

	   where the container is parameterized to hold Serializables.

	   Provided that the contained type(s) conform to
	   cleanup_ptr's requirements, this will recursively clean up
	   sub-sub-...subcontainers.

	   Note that Serializable containers should have a cleanup
	   functor installed as part of their registration, making
	   this class unnecessary for most cases: simply calling
	   cleanup_serializable() will recursively walk/clean such
	   containers. The underlying cleanup algos might use this
	   type, however (at least one of them does).

	   Added in 1.1.3.
	 */
        struct cleaner_upper
        {
                /**
                   Calls cleanup_serializable<T>(t)
                */
                template <typename T>
                void operator()( T & t ) throw()
                {
                        cleanup_serializable<T>( t );
                }
                /**
                   Calls cleanup_serializable<T>(t).
                */
                template <typename T>
                void operator()( T * & t ) throw()
                {
                        cleanup_serializable<T>( t );
                }
        };


	/**
	   An auto_ptr-like type intended to simplify
	   pointer/exception safety in some deserialization algorithms
	   by providing a way to completely and safely destroy
	   partially-deserialized objects.

	   SerializableT must either have an explicit s11n_traits
	   specialization installed or work properly with the default
	   functor provided by s11n_traits::cleanup_functor.  In
	   practice, this means that types which manage the memory of
	   their contained pointers are safe to work with the default,
	   whereas the cleanup of unmanaged child pointers (e.g., std
	   containers) requires a proper specialization.

	   Note that this type does not have copy/assignment ctors,
	   due to the conventional constness of their right-hand
	   sides: use the swap() or take() members to take over a
	   pointer.

	   Added in 1.1.3.
	*/
	template <typename SerializableT>
	struct cleanup_ptr
	{
	public:
		typedef SerializableT cleaned_type;
	private:
		cleaned_type * m_ptr;
		cleanup_ptr & operator=( const cleanup_ptr & ); // Not Implemented
		cleanup_ptr( const cleanup_ptr & ); // Not Implemented
		void cleanup() throw()
		{
			if( this->m_ptr )
			{
				cleanup_serializable<cleaned_type>( this->m_ptr );
			}
		}
	public:
		/**
		   Constructs an object pointing to nothing.
		*/
		cleanup_ptr() throw() : m_ptr(0) 
		{
		}
		/**
		   Transfers ownership of p to this object.
		 */
		cleanup_ptr( cleaned_type * p ) throw() : m_ptr(p)
		{
		}

		/**
		   Uses s11n::cleanup_serializable<cleaned_type>()
		   to free up up this->get().
		*/
		~cleanup_ptr() throw()
		{
			this->cleanup();
		}
		/**
		   Dereferences this object's pointed-to object.  If
		   this object does not point to anything it throws a
		   std::runtime_error with an informative what()
		   message explaining the error.
		 */
		cleaned_type & operator*()
		{
			if( ! this->m_ptr )
			{
				throw std::runtime_error("Attempt to dereference a null pointer via s11n::cleanup_ptr<>::operator*()" );
			}
			return *this->m_ptr;
		}

		/**
		   Returns the same as get().
		*/
		cleaned_type * operator->() throw()
		{
			return this->m_ptr;
		}

		/**
		   Returns this object's pointed-to object without
		   transfering ownership.
		*/
		cleaned_type * get() throw()
		{
			return this->m_ptr;
		}

		/**
		   Transfers ownership of p to this object. This
		   member takes the place of copy/assign operators,
		   since those conventionally take a const right-hand
		   argument.

		   Destroys the object this object pointed to before
		   taking over ownership. 0 is a legal value for p.

		   If (p == this->get()) then this function does
		   nothing.

		   Postcondition: p == this->get() 
		*/
		void take( cleaned_type * p ) throw()
		{
			if( p != this->m_ptr )
			{
				this->cleanup();
				this->m_ptr = p;
			}
		}

		/**
		   Transfers ownership of this->get() to the
		   caller.

		   Postcondition: 0 == this->get() 
		*/
		cleaned_type * release() throw()
		{
			cleaned_type * x = this->m_ptr;
			this->m_ptr = 0;
			return x;
		}

		/**
		   Cleans up any pointed-to object and points this
		   object at 0. Does nothing if this object points
		   to no object.

		   Postcondition: 0 == this->get() 
		*/
		void clean() throw()
		{
			this->take( 0 );
		}

		/**
		   Swaps ownership of pointers with rhs.
		*/
		void swap( cleanup_ptr & rhs ) throw()
		{
			cleaned_type * x = this->m_ptr;
			this->m_ptr = rhs.m_ptr;
			rhs.m_ptr = x;
		}
	};
	

        /**
           Deserializes target from src using the default API marshaling
           mechanism. Returns true on success.

	   On error it returns false or passes on an exception. In
	   either case, target might be in an undefined state, and may
	   need manual interaction to free up resources (e.g., a list
	   of pointers might have some pointers which need to be
	   cleaned up during exception handling). The exact definition
	   of its state after a failure is specified by the algorithm
	   which deserializes the target (as defined via
	   s11n_traits<DeserializableT>::deserialize_functor).
        */
        template <typename DataNodeType, typename DeserializableT>
        bool deserialize( const DataNodeType & src, DeserializableT & target );


	/**
	   Like the standard form of deserialize(), but if passed a
	   null pointer, it attempts to classload the class and assign
	   the passed-in pointer to it. If passed a non-null target
	   then it behaves as if target were a reference, simply
	   passing on the target after dereferencing it.

	   For example:

<pre>
T * t = 0;
deserialize<NodeType,T>( mynode, t );
// t will be non-0 if it worked.

T * x = new X;
if( deserialize<NodeType,T>( mynode, x ) )
{
  // x is now populated exactly as if we had called:
  // deserialize<NodeType,T>( mynode, *x );
}
</pre>

           To get the class name, the algorithm first tries
           node_traits<DataNodeType>::class_name(src). If it cannot
           load a class using that name, it tries
           s11n_traits<DeserializableT>::class_name(target).

           Underlying calls to s11n::cl::classload() and serialization
           proxies may throw. If they do, the exception is passed on
	   to the caller.

	   If passed a null pointer and this function fails, target is
	   not modified. If deserialization fails, any
	   internally-created (DeserializableT*) is deallocated using
	   cleanup_serializable(). If passed a non-null pointer and
	   the function fails, behaviour is as for the non-pointer
	   variant of deserialize() - target may or may not be in a
	   defined state, as defined by the specific proxy algorithm.


	   Added in s11n version 1.1.3.

	*/
        template <typename DataNodeType, typename DeserializableT>
	bool deserialize( const DataNodeType & src, DeserializableT * & target );

	/**
	   Identical to deserialize(const
	   DataNodeType&,DeserializableT*&) except that it works on a
	   cleanup_ptr<>. The target may be empty (pointing to zero):
	   if it is then dynamic loading is attempted, as described
	   in the docs for the non-cleanup_ptr variant of this
	   function.

	   Returns true if deserialization to the target succeeds,
	   else false.  If it returns false, target.get() still points
	   to the same object it did when function was called (which
	   may be 0). Whether or not the contained object is modified
	   on deserialization failure depends on the underlying
	   algorithm used to deserialize it.
	*/
        template <typename DataNodeType, typename DeserializableT>
	bool deserialize( const DataNodeType & src, cleanup_ptr<DeserializableT> & target );

        /**
           Tries to deserialize a DeserializableT from src, using
           <code>s11n_traits<DeserializableT>::factory_type()(node_traits<DataNodeType>::class_name(src))</code>
           to create a new DeserializableT. 

	   DeserializableT may not be a pointer-qualified type.

	   On error it returns 0 or propagates on an exception,
	   otherwise returns a pointer to a new object, which the
	   caller takes ownership of.

	   As of 1.1.3, this function relies on
	   s11n::cleanup_serializable() in order to be able to specify
	   what happens to the internally allocated object if
	   deserialization fails. That function is called on the
	   object before this function returns if deserialization
	   fails.

	   Prior to 1.1.3 this function would leak if this function
	   failed and if DeserializableT contained unmanaged pointers
	   (even indirectly, via sub-containment), such as in
	   list<int*> or list<map<int,string*>>.
        */
        template <typename DataNodeType, typename DeserializableT>
        DeserializableT * deserialize( const DataNodeType & src );


        /**
           Clones an arbitrary SerializableType using its
           DataNodeType serialization implementation.

           Returns a clone of tocp, or returns 0 on error.
           The caller owns the returned pointer.

           This copy is polymorphism-safe as long as all participating
           Serializables (re)implement the appropriate de/serialize
           operations, similarly to as they would do for a copy ctor
           or classical Clone() member function.

           Tip: s11n_clone() is a convenient way to test new
           de/serialize functions, e.g., for new Serializables,
           because if it works then deserializing from streams/files
           will also work. This function takes SerializableType
           through the whole de/serialize process except for i/o,
           including classloading.

	   This function was renamed from clone() in version 1.1.

	   Exceptions and errors:

	   This function may return 0 or throw on an error, as
	   dictated by serialize() and then deserialize() (in that
	   order). Thus safety guarantees are defined in terms
	   of those operations.
        */
        template <typename DataNodeType, typename SerializableType>
        SerializableType * s11n_clone( const SerializableType & tocp );

        /**
           "Casts" t1 to t2 using serialization. This will work
           whenever t1 and t2 are "semantically compatible", whatever
           that really means. It can be used, e.g., to copy a
           list<int> to a vector<double>, provided both
           types have been proxied. In practice, this means: if Type1
           and Type2 both use the same, or compatible, de/ser
           algorithms, they are s11n_cast-able to one another.

           Note that in the case of containers, the pointerness of the
           contained types is irrelevant: this works on both, thus
           a list<int> can be "cast" to a vector<double*>.

           As usual for a failed deserialization, if it returns false
           then t2 may be in an undefined state. There is no guaranty,
           however, that t2's deserialize operator will ever be
           called, as the serialization of t1 must first succeed
           for that to happen.

	   Type2 may not currently be a pointer type, but Type1 may
	   be. This will be fixed someday (when someone complains).

	   Exceptions and errors:

	   On error this function will return false or propagate
	   an exception, as dictated by serialize() and then
	   deserialize() (in that order).

	   If Type1 and Type2 are not guaranteed to be monomorphic or
	   base-most Serializable types, then it is good practice to
	   explicitely specify them as templatized parameters, and not
	   rely on implicit type selection, which might choose the
	   wrong type (not the base-most one, which is what s11n is
	   "keyed" to), which will mean that s11n "can't find" the
	   registration code for the type.
        */
        template <typename NodeType, typename Type1, typename Type2>
        bool s11n_cast( const Type1 & t1, Type2 & t2 );


} // namespace s11n

#include <s11n.net/s11n/serialize.tpp> // implementations for above-declared code


#endif // s11n_SERIALIZE_HPP_INCLUDED
