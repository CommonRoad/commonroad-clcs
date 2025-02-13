#ifndef s11n_net_s11n_v1_1_ALGO_HPP_INCLUDED
#define s11n_net_s11n_v1_1_ALGO_HPP_INCLUDED 1
/////////////////////////////////////////////////////////////////////////
// algo.hpp: generic functors and algos for use with libs11n.
// Author: stephan beal <stephan@s11n.net>
// License: Public Domain
/////////////////////////////////////////////////////////////////////////

#include <string>
#include <algorithm>

#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // COUT/CERR
#include <s11n.net/s11n/serialize.hpp> // core de/serialize interface
#include <s11n.net/s11n/traits.hpp> // node_traits
#include <s11n.net/s11n/type_traits.hpp> // type_traits<>

namespace s11n
{

        /**
           Serializes src to as a subnode of target, named
           nodename. Except for the addition of a subnode, it is
           identical to serialize( target, src ).

	   If serialization into the subnode throws the subnode is not
	   added to target (it is destroyed) and any exception is
	   propagated back to the caller.

           This is a convenience function: not part of the s11n kernel.


	   Changed in 1.1.3:

	   - Moved to algo.hpp to be symetric with deserialize_subnode().

        */
        template <typename DataNodeType, typename SerializableT >
        bool serialize_subnode( DataNodeType & target,
                                const std::string & nodename,
                                const SerializableT & src );


        /**
           If a child named subnodename is found in src then this function
           returns deserialize( child, target ) and returns it's result, otherwise
           it returns 0.

	   The function might throw, as it uses the two-arg form of
	   deserialize().

           This is a convenience function: not part of the s11n
           kernel.


	   Changed in 1.1.3:

	   - Moved to algo.hpp to avoid a circular dependency on
	   s11n::find_child_by_name().

        */
        template <typename DataNodeType, typename DeserializableT>
        bool deserialize_subnode( const DataNodeType & src,
                                  const std::string & subnodename,
                                  DeserializableT & target );

        /**
           If a child named subnodename is found in src then this function
           returns the result of deserialize(child), otherwise
           it returns 0.

	   The function might throw, as it uses the two-arg form of
	   deserialize().

           This is a convenience function: not part of the s11n kernel.

	   Changed in 1.1.3:

	   - Moved to algo.hpp to avoid a circular dependency on
	   s11n::find_child_by_name().

        */
        template <typename DataNodeType, typename DeserializableT>
        DeserializableT * deserialize_subnode( const DataNodeType & src,
                                               const std::string & subnodename );



        /**
           For each item in [first,last), copies the item to OutputIt
           if pred(*item) returns true.

           Code copied from:

           http://www.bauklimatik-dresden.de/privat/nicolai/html/en/cpp.html
        */

        template <typename InputIt, typename OutputIt, typename UnaryPredicate>
        OutputIt copy_if (InputIt first,
                          InputIt last,
                          OutputIt result,
                          UnaryPredicate pred) 
        {
                while (first != last)
                {
                        if (pred(*first)) *result++ = *first;
                        ++first;
                }
                return result;
        }

 

        /**
           Deletes an object passed to it.

           This type accepts non-pointer types for "destruction." This
           is a non-operation, and is supported to allow other
           template code to generically free objects without needing
           to know if they are pointers. This allows some
           formerly-non-consolidatable reference-vs-pointer-type code to
           share a single implementation, as "delete(myobj)" is not
           valid for non-pointer types, but object_deleter()(myobj)
           is.

	   DEPRECATED: this will be removed in favor of
	   cleanup_serializable().

	   @deprecated
        */
        struct object_deleter
        {
                /**
                   Deletes t.
                */
                template <typename T>
                void operator()( T * t ) const
                {
                        // ACH!!!! If we use (const T *) for the arguments
                        // then the (const T &) version is called
                        // even when a pointer type is passed in!!!!

                        // i don't fully understand why
                        // delete( const T * ) is legal, 
                        // considering that it triggers a dtor,
                        // and dtors are non-const.
                        //CERR << "object_deleter deleting "<< std::hex<<t<<"\n";
                        delete( t );
                }
                /**
                   Does nothing: is here to allow some
                   reference-vs-pointer-type transparency.
                */
                template <typename T>
                void operator()( const T & t ) const
                {
                        //CERR << "object_deleter no-op\n";
                }
        };



        /**
           For each item in [begin,end) object_deleter()(*item) is called.

           After this call the container from which the iterators come
           still contains the items but they are invalid - deleted
           pointers. The client should call erase(begin,end) to delete
           the entries, or use convenience function
           free_list_entries(), which accepts a list-style
           container and empties the list.

	   DEPRECATED: this will be removed in favor of
	   cleanup_serializable().

	   @deprecated
        */
        template <typename IterT>
        void delete_objects( IterT begin, IterT end )
        {
                std::for_each( begin, end, object_deleter() );
        }



        /**
           object_reference_wrapper is a type for giving access
           to T objects via their dot operator, regardless of whether
           they are pointers or not.

           Intended for use with value_types which come from, e.g.,
           std::list, so objects of those types can be called using
           the same syntax regardless of whether they are pointer
           types or not.

           e.g., assuming MyType might be a pointer or a reference,
           we can ignore that difference for call-syntax purposes
           with:

<pre>           
object_reference_wrapper<MyType> wrap;
wrap(myobj).memberfunc();
</pre>

or:

<pre>
object_reference_wrapper<MyType> wrap(myobj);
wrap().memberfunc();           
</pre>
        */
        template <typename T>
        struct object_reference_wrapper
        {
                typedef T value_type;
                typedef T base_value_type;
//                 object_reference_wrapper() : m_ptr(0) {}
                object_reference_wrapper( value_type &obj ) : m_ptr(&obj) {};
                /**
                   Sets this object's proxy object to t and returns t.
                 */
                base_value_type & operator()( value_type & t )
                {
                        return this->m_ptr = &t;
                        return t;
                }
                /**
                   Returns this object's wrapped object.
                 */
                base_value_type & operator()() const { return *(this->m_ptr); }

                /**
                   Returns true if this object is wrapping a non-0 object, else
                   false.
                */
                bool good() const
                {
                        return 0 != this->m_ptr;
                }

        private:
                value_type * m_ptr;
        };

        /**
           A specialization to wrap pointers to (T *) such that they
           can be accessed, via this wrapper, using a dot instead of
           <tt>-></tt>.
        */
        template <typename T>
        struct object_reference_wrapper<T *>
        {
                typedef T * value_type;
                typedef T base_value_type;
//                 object_reference_wrapper() : m_ptr(0) {}
                object_reference_wrapper( value_type & obj ) : m_ptr(obj) {};
                /** Sets this object's proxied object to t and Returns t. */
                base_value_type & operator()( value_type & t )
                {
                        this->m_ptr = &t;
                        return *t;
                }
                /** Returns this object's wrapped object.
                */
                base_value_type & operator()() const { return *(this->m_ptr); }
        private:
                base_value_type * m_ptr;
        };


        /**
           const_object_reference_wrapper is identical in usage to
           object_reference_wrapper, except that it deals with const
           objects. It is a separate functor to avoid ambiguity and
           some impossible overloads.
        */
        template <typename T>
        struct const_object_reference_wrapper
        {
                typedef T value_type;
                typedef T base_value_type;
//                 const_object_reference_wrapper() : m_ptr(0) {}
                const_object_reference_wrapper( const value_type &obj ) : m_ptr(&obj) {};
                /**
                   Sets this object's proxied obj to t and returns t.
                 */
                const base_value_type & operator()( const value_type & t )
                {
                        this->m_ptr = &t;
                        return t;
                }
                /**
                   Returns this object's wrapped object.
                 */
                const base_value_type & operator()() const { return *this->m_ptr; }
        private:
                const value_type * m_ptr;
        };

        /**
           A specialization to wrap pointers to (T *) such that they
           can be accessed, via this wrapper, using a dot instead of
           <tt>-></tt>.
        */
        template <typename T>
        struct const_object_reference_wrapper<T *>
        {
                typedef T * value_type;
                typedef T base_value_type;
//                 const_object_reference_wrapper() : m_ptr(0) {}
                explicit const_object_reference_wrapper( const value_type & obj ) : m_ptr(obj) {};
                /** Returns (*t). */
                inline const base_value_type & operator()( value_type & t )
                { 
                        this->m_ptr = &t;
			return t;
                }
                /** Returns this object's wrapped object. It does not check for validity.*/
                inline const base_value_type & operator()() const { return *(this->m_ptr); }
        private:
                const base_value_type * m_ptr;
        };


	namespace Detail {
		/**
		   child_pointer_deep_copier is a functor to deep-copy
		   a list of pointers into another list. Designed for
		   use with std::for_each and the like.
		   
		   Assuming T is the type contained in ListType, stripped
		   of any pointer part, then the following must hold:
		   

		   - List must support <code>push_back( T * )</code>.

		   - This must be a valid expression:

		   <code>T * t = new T( *p );</code>
	   
		   Where p is a passed to this type's operator().

		   ACHTUNG: This is only useful for non-polymorphic
		   copying.

		   It might be interesting to note that copying
		   monomorphic s11n::s11n_node objects this way is
		   "pseudo-polymorphic" - the copy itself is
		   monomorphic but the data needed to deserialize the
		   proper type from the node is maintained.
		*/
		template <typename ListType>
		class child_pointer_deep_copier
		{
		public:
			typedef ListType list_type;
			typedef typename ListType::value_type full_value_type;
			typedef typename ::s11n::type_traits<full_value_type>::type value_type; // list_type::value_type minus any pointer part.
			/**
			   Target list must outlive this object.
			*/
			child_pointer_deep_copier( list_type & target ) : m_childs(&target)
			{}
			
			/**
			   Inserts a copy of p into this object's list and returns true.
			   
			   Returns true if p is successfully copied.
			   
			   If an exception thrown while copying, this function
			   will catch it and not modify the underlying
			   container. In that case, false is returned.
			   
			   The target list takes ownership of the new copy of p.
			*/
			bool operator()( const value_type * p ) throw()
			{
				if( ! this->m_childs || ! p ) return false;
				value_type * cp = 0;
				try
				{
					cp = new value_type( *p );
					if( ! cp ) return false;
				}
				catch(...)
				{
					delete( cp ); // not necessary
					return false;
				}
				this->m_childs->push_back( cp );
				return true;
			}
		private:
			list_type * m_childs;
		};


		/**
		   Functor to return true if given NodeT objects match
		   a certain name. NodeT must be compatible with
		   node_traits<NodeT>.

		   This class is admittedly to avoid the use of bind1st/bind2nd
		   :/.
		*/
		template <typename NodeT>
		struct same_name
		{
			typedef NodeT node_type;
			typedef ::s11n::node_traits< NodeT > traits_t;
			explicit same_name( const std::string & n ) : m_name(n)
			{}
			
			inline bool operator()( const node_type * x ) const
			{
				return ( ! x  )
					? this->m_name.empty()
					: (traits_t::name( *x ) == this->m_name);
			}
		private:
			std::string m_name;
		};

	} // namespace Detail



        /**
           A helper functor to loop over serializable
           children.

           Designed for use with std::for_each().

           NodeType must be compatible with s11n node conventions.

           Please see the operator() docs for important usage
           information, especially if you want to use this
           object outside the context of for_each().
        */
        template <typename NodeType>
        struct subnode_serializer_f
        {
                typedef NodeType node_type;
                /**
                   Creates an object for serializing

                   Preconditions:

                   - dest must outlive this object.
                   More correctly, this object's
                   operator() must not be called after
                   dest is destroyed.

                */
                subnode_serializer_f( node_type & dest, const std::string & subname )
                        : result(true), m_root(&dest), m_name(subname)
                {
                }
 
                /**
                   Serializes src into a subnode of dest using the
                   name given in this object's ctor.

                   Note that during an, e.g., for_each() this object
                   will return false on a failed serialize, and will
                   CONTINUE to return false on additional serializations.
                   This is to avoid the possibility that for_each()
                   fails on the first item of a list, handles 3000 items,
                   and then the whole thing fails because of the first one.
                   Thus, this operator will never process another request
                   once it has returned false ONCE.

		   Versions prior to 1.1.3 always dereferenced src,
		   without checking for a null pointer. As of 1.1.3,
		   false is returned if (!src).
                */
                template <typename SerializableT>
                inline bool operator()( const SerializableT * src )
                {
                        return (src && this->result)
				? (this->result = ::s11n::serialize_subnode( *this->m_root,
									      this->m_name,
									      *src ))
				: false;
                }
                template <typename SerializableT>
                inline bool operator()( const SerializableT & src )
                {
                        return this->operator()( &src );
                }

                /**
                   For use as a "return value catcher" for std::for_each().
                   See operator() for how it is set. The starting value
                   is true, which means that looping over an empty list
                   with this object will return a true result (which is
                   the convention in s11n).
                */
                bool result;
        private:
                node_type * m_root;
                std::string m_name;
        };




        /**
           A Serializable Proxy for streamable types. It "should" work
           with any type which meets these conditions:

           - complementary i/ostream operators are implemented (as
           member or free functions).

           - supports a copy ctor (for deserialization).


	   The class name of serialized objects will be taken
	   from s11n_traits<>::class_name(), which isn't truly
	   valid because most streamable types are never registed
	   via s11n_traits. Hmmm.

           Its output is significantly bigger than using, e.g. node
           properties to store them, but with this proxy any
           streamable can be treated as a full-fledged Serializable,
           which allows some generic container-based serialization to
           be done regardless of the underlying types (see the various
           standard container algos for examples).

	   ACHTUNG: never pass the same Serializable to the
	   operators more than once or you will get duplicate and/or
	   incorrect data.
        */
        struct streamable_type_serialization_proxy
        {
                /**
                */
                streamable_type_serialization_proxy()
                {}

                /**
                   Creates a property in dest, called 'v', and sets
                   its value to src using node_traits<NodeType>::set(dest,"v",src).

                   Always returns true unless no class name can be found
		   for src, in which case it returns false to avoid inserting
		   a non-name node into the data tree (which might
		   result in unreadable data later).
                */
                template <typename NodeType, typename SerType>
                bool operator()( NodeType & dest, const SerType & src ) const
                {
                        typedef node_traits<NodeType> NTR;
			typedef s11n_traits<SerType> STR;
                        NTR::class_name( dest, STR::class_name(&src) );
                        NTR::set( dest, "v", src );
                        return true;
                }

                /**
                   Looks for a property set by the serialize operator and sets
                   dest to its value. The default for dest, in the case
                   of a missing property or nonconvertable value is
                   dest itself.

                   If the property is missing this function throws an
                   s11n_exception. [Design note: this seems a bit
                   harsh.]

                   On success dest gets assigned the property's value
                   and true is returned. This function cannot catch a
                   case of inability to convert the value into a
                   SerType: client code interested in doing so should
                   compare dest's value to a known error value after
                   this function returns or throw from that type's
                   istream operator on error.
                */
                template <typename NodeType, typename SerType>
                bool operator()( const NodeType & src, SerType & dest ) const
                {
                        typedef node_traits<NodeType> NTR;
                        if( ! NTR::is_set( src, "v" ) )
                        {
				throw s11n_exception( "streamable_serializable_proxy: deser failed: property 'v' missing!" );
                        }
                        dest = NTR::get( src, "v", SerType() /* should never happen */ );
                        return true;
                }
        };


        /**
           Adds ch as a child of parent. Parent takes over ownership
           of ch.

           NodeType must have a node_traits<> specialization.
        */
        template <typename NodeType, typename ChildType>
        inline void add_child( NodeType & parent, ChildType * ch )
        {
                typedef ::s11n::node_traits<NodeType> NTR;
                NTR::children( parent ).push_back( ch );
        }


        /**
           Creates a new node, named nodename, as a child of parent.

           Returns a reference to the new child, which parent now
           owns.

           NodeType must have a node_traits<> specialization or work
           using the default.

	   Development tip: this function often comes in handy
	   during serialization.
        */
        template <typename NodeType>
        NodeType & create_child( NodeType & parent, const std::string nodename );


        /**
           Each child in parent.children() which has the given name is
           copied into the target container.

           Returns the number of items added to target.

           DestContainerT must support an insert iterator which will
           insert the pointer type contained in the list returned by
           parent.children(). i.e., it must hold (const
           NodeT *).

           Ownership of the children do not change by calling this
           function. Normally they are owned by the parent node
           (unless the client explicitely does something to change
           that).
        */
        template <typename NodeT, typename DestContainerT>
        size_t find_children_by_name( const NodeT & parent, const std::string & name, DestContainerT & target );

        /**
           Finds the FIRST child in parent with the given name and
           returns a pointer to it, or 0 if no such child is found.

           Ownership of the child does not change by calling this
           function: parent still owns it.

	   Complexity is linear.
        */
        template <typename NodeT>
        const NodeT *
        find_child_by_name( const NodeT & parent, const std::string & name );

        /**
           A non-const overload of find_child_by_name(). Functionally
           identical to the const form, except for the constness of
           the parent argument and return value.

           Ownership of the returned pointer is not changed by calling
           this function (normally parent owns it, but clients may
           change that without affecting this function). When in
           doubt, i.e. during "normal usage", do NOT delete the returned
           pointer, because the parent node owns it. This function can
	   be used to find a child for manual removal from parent via
	   the API for the node_traits<NodeT>::children(parent) object.
        */
        template <typename NodeT>
        NodeT *
        find_child_by_name( NodeT & parent, const std::string & name );

	namespace debug {

		/**
		   Dumps a tree-like view of n's structure, excluding properties,
		   to cerr. The second parameter is for use by this function
		   in recursion: do not pass a value for it.
		*/
		template <typename NodeT>
		void dump_node_structure( const NodeT & n, int indentlv = 0 );
	} // namespace




	/**
	   A functor which simply forwards its arguments to
	   s11n::serialize_subnode().

	   Added in 1.1.3.
	*/
	struct serialize_subnode_f
	{
		template <typename NT, typename ST>
		inline bool operator()( NT & dest, const std::string & subname, const ST & src ) const
		{
			return serialize_subnode<NT,ST>( dest, subname, src );
		}
	};

 	/**
	   A functor which simply forwards its arguments to
	   s11n::deserialize_subnode().

	   Added in 1.1.3.
	*/
	struct deserialize_subnode_f
	{
		template <typename NT, typename ST>
		inline bool operator()( NT & dest, const std::string & subname, const ST & src ) const
		{
			return deserialize_subnode<NT,ST>( dest, subname, src );
		}
	};


    /**
       This is similar to serialize() but adds a version tag to the
       serialized data. To deserialize you must use
       deserialize_versioned(), passing it the same version.

       VersionType must be lexically castable (i.e. i/o-streamable),
       must be comparable for equivalence, and is expected to be a
       numeric or std::string type, or something similar.

       This version is stored as a property (called "version") of dest,
       and the src object is stored in a sub-node of dest (named
       "vdata").

       On error it returns false or propagates an exception. On
       success it returns true.

       Added in 1.3.1 + 1.2.7.
    */
    template <typename NodeT, typename VersionType, typename SerT>
    bool serialize_versioned( NodeT & dest, VersionType const ver, SerT const & src );

    /**
       This is the counterpart to serialize_versioned(). If src
       contains a property named "version" which lexically matches ver
       then a sub-node of src (named "vdata") containing serialized data is
       used to deserialize the dest object.

       On a version mismatch, or if src does not contain the expected
       child node then this function throws an s11n_exception. If
       deserialization to dest fails then false might be returned or
       an exception may be propagated.

       Mis-feature: you cannot pass a (char const *) (i.e. a string
       literal) as the version object because that causes lots of
       ambiguity errors in the lexical casting proces. You may pass
       std::string() objects, however.

       Added in 1.3.1 + 1.2.7.
    */
    template <typename NodeType, typename VersionType, typename SerType>
    bool deserialize_versioned( NodeType const & src, VersionType const ver, SerType & dest );

} // namespace


#include <s11n.net/s11n/algo.tpp> // implementations
#endif // s11n_net_s11n_v1_1_ALGO_HPP_INCLUDED
