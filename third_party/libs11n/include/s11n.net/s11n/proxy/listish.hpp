////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
#ifndef s11n_net_s11n_v1_1_LIST_HPP_INCLUDED
#define s11n_net_s11n_v1_1_LIST_HPP_INCLUDED 1


namespace s11n {


	/**
           The s11n::list namespace defines functors and algorithms for
           working with std::list/vector-style containers.
        */
        namespace list {

                /**
                   serialize_list() supports list/vector-like types containing
                   any Serializable type. Serializes (src.begin(),src.end()]
                   to dest. Each item has a node name of that given to the
                   ctor or an unspecified dummy value.
           
                   Returns true on success. If false is returned then dest is
                   in an undefined state: some number of serializations may
                   have succeeded before the failure. This operation stops
                   processing at the first serialization error.

		   If serialization of a child fails, the child is not
		   added to dest and any exception is propagated back
		   to the caller.

                   Compatible ListTypes must support:
           
                   - a value_type typedef describing the type of it's contents.

                   - push_back( value_type )

                   - const_iterator typedefs.

                   Restrictions on value_type's type:

                   - Must be a Serializable. This includes any i/o streamable
                   type which has a proxy installed by s11n (includes all PODs
                   and std::string by default).

                   - May be a pointer type. NodeType must not be a pointer.

                   Some s11n-side requirements:

                   - ListType must be registered with the ListType classloader.

		   If the underlying call to ::s11n::serialize()
		   throws then the exception is propagated. If it
		   throws, then dest is left in an undefined state,
		   but no memory is leaked here because NodeType is
		   responsible for cleaning up any of its children.

                   ACHTUNG: never pass the same destination container to 
                   this function more than once or you will get duplicate and/or
                   incorrect data.

		   As of version 1.1.3, this function throws an s11n_exception
		   if dest is not empty. The reason for this is to enforce
		   that clients do not accidentally re-use the same (populated)
		   node for serialization of multiple objects, which would cause
		   deserialization of the container to fail.
                */
                template <typename NodeType, typename SerType>
                bool serialize_list( NodeType & dest, const SerType & src );



                /**
                   Identical to the two-argument form of serialize_list(), but
                   serializes src into a subnode of dest, named subnodename.

		   If serialization into dest child fails, the child
		   node is not added to dest and the error (possibly
		   an exception) is propagated back to the caller.
                */
                template <typename NodeType, typename SerType>
                bool serialize_list( NodeType & dest, const std::string & subnodename, const SerType & src );


		/**
		   Functor equivalent of serialize_list().

		   Added in version 1.1.3.
		*/
                struct serialize_list_f : ::s11n::serialize_binary_f_tag
		{
			template <typename NodeType, typename SerType>
			inline bool operator()( NodeType & dest, const SerType & src ) const
			{
				return serialize_list<NodeType,SerType>( dest, src );
			}

			template <typename NodeType, typename SerType>
			inline bool operator()( NodeType & dest, const std::string & subnodename, const SerType & src ) const
			{
				return serialize_list<NodeType,SerType>( dest, src );
			}
		};


                /**
                   For each [src.children().begin(),end()) an object
                   of type SerType::value_type is created, deserialized,
                   and is added to dest via push_back( item ).

                   See serialize_list() for the list type requirements.

                   If SerType::value_type is a pointer type then dest owns any
                   newly-created pointers, and it's owner (or the container
                   itself) is responsible for cleaning them up. (Non-pointer
                   types need no explicit clean-up, of course.)

		   SerType requirements:

		   - Must be a Serializable type, as must contained type(s).

		   - May not be a pointer type, but may contain pointer types.

		   - A working s11n_traits<SerType>::cleanup_functor must be
		   installed which properly frees up list entries.

		   - Must have a swap() member function, to support the
		   no-change-on-error policy.

		   dest must be empty upon calling this function, or
		   its contents will be irrevocably lost when this
		   function succeeds (which would lead to a leak if
		   dest contained unmanaged pointers, even
		   indirectly).


                   Returns true if all deserializations succeed.
                   Stops processing and returns false at the first
                   error, in which case dest is not modified: some
                   children may or may not have been successfully
                   deserialized, but we destroy them if
                   deserialization of any fail. One immediate
                   implication of this is that it will fail if src
                   contains any other children than the type which
                   dest expects. The s11n_traits::cleanup_functor
		   is used to ensure that cleanup walks through
		   any containers which hold pointers, deallocating
		   them as well.

		   This function only throws if an underlying call to
		   deserialize() or the classloader throws or if there
		   is some sort of internal error (e.g., src contains
		   child pointers which point to NULL).

		   Major behaviour change in 1.1.3+:

		   If this function throws or returns false then the
		   target list is left untouched. See
		   ::s11n::map::deserialize_map() for more
		   information.
                */
                template <typename NodeType, typename SerType>
                bool deserialize_list( const NodeType & src, SerType & dest );



                /**
                   Identical to the two-argument form of deserialize_list(), but
                   deserializes a subnode of src, named subnodename. If no such
                   child is found in src then false is returned.
                */
                template <typename NodeType, typename SerType>
                bool deserialize_list( const NodeType & src, const std::string & subnodename, SerType & dest );


		/**
		   Functor equivalent of deserialize_list().

		   Added in version 1.1.3.
		*/
                struct deserialize_list_f : ::s11n::deserialize_binary_f_tag
		{

			template <typename NodeType, typename SerType>
			inline bool operator()( const NodeType & src, SerType & dest ) const
			{
				return deserialize_list<NodeType,SerType>( src, dest );
			}

			template <typename NodeType, typename SerType>
			inline bool operator()( const NodeType & src, const std::string & subnodename, SerType & dest ) const
			{
				return deserialize_list<NodeType,SerType>( src, dest );
			}
		};


                /**
                   serialize_streamable_list serializes objects of type
                   <tt>std::list&lt;X&gt;</tt> (and compatible list types,
                   such as std::vector). It stores them in such a way that
                   they can be loaded into any compatible container via
                   deserialize_streamable_list().

                   Conventions:

                   - NodeType must support a generic set(Key,Val) function, as
                   implemented by the s11n::data_node interface.

                   - ListType must conform to std::list conventions and it's
                   value_type must be a non-pointer type which is
                   i/ostreamable (this includes all PODs and
                   std::string). Pointers are not supported by
                   this function.

                   ACHTUNG:

                   - Never call this on a node for which you store other
                   properties, as deserialize_streamable_list() will consume them
                   if you use that function.

                   - This function sets dummy property keys, both to please
                   the conventions of keys having non-empty values and to keep
                   the list in the proper order. It uses keys which should be
                   portable to, e.g., standard XML.

                   Always returns true - the bool return value is for API consistency.

                   ACHTUNG 2:

                   - The return type of this function was changed from size_t
                   to bool in version 0.9.10.
                */
                template <typename NodeType, typename ListType>
                bool serialize_streamable_list( NodeType & dest, const ListType & src );


                /**
                   Identical to serialize_streamable_list(dest,src), but
                   creates a subnode in dest, named subnodename, where the
                   data is stored.

                   ACHTUNG:

                   - The return type of this function was changed from size_t
                   to bool in version 0.9.10.
                */
                template <typename NodeType, typename ListType>
                bool serialize_streamable_list( NodeType & dest, const std::string & subnodename, const ListType & src );


		/**
		   Functor equivalent of serialize_streamable_list().

		   Added in version 1.1.3.
		*/
                struct serialize_streamable_list_f : ::s11n::serialize_binary_f_tag
		{
			template <typename NodeType, typename ListType>
			inline bool operator()( NodeType & dest, const ListType & src ) const
			{
				return serialize_streamable_list<NodeType,ListType>( dest, src );
			}

			template <typename NodeType, typename ListType>
			inline bool operator()( NodeType & dest, const std::string & subnodename, const ListType & src ) const
			{
				return serialize_streamable_list<NodeType,ListType>( dest, src );
			}
		};


                /**
                   Deserializes dest from src. It reads in all properties from
                   src, ignoring their keys and taking only their values. This
                   is suitable for use with the result of a
                   serialize_streamable_list() call. See that function for more
                   information, including the conventions which must be
                   supported by NodeType and ListType.
           
                   Always returns true - the bool return value is for API consistency.

                   ACHTUNG:

                   - The return type of this function was changed from size_t
                   to bool in version 0.9.10.

                */
                template <typename NodeType, typename ListType>
                bool deserialize_streamable_list( const NodeType & src, ListType & dest );



                /**
                   Identical to deserialize_streamable_list(), but looks for
                   the data in a subnode of src named subnodename.

                   Returns false if no child could be found.

                   ACHTUNG:

                   - The return type of this function was changed from size_t
                   to bool in version 0.9.10.

                */
                template <typename NodeType, typename ListType>
                bool deserialize_streamable_list( const NodeType & src, const std::string & subnodename, ListType & dest );





		/**
		   Functor equivalent of deserialize_streamable_list().

		   Added in version 1.1.3.
		*/
                struct deserialize_streamable_list_f : ::s11n::deserialize_binary_f_tag
		{
			template <typename NodeType, typename ListType>
			inline bool operator()( const NodeType & src, const std::string & subnodename, ListType & dest ) const
			{
				return deserialize_streamable_list<NodeType,ListType>( src, dest );
			}

			template <typename NodeType, typename ListType>
			inline bool operator()( const NodeType & src, ListType & dest ) const
			{
				return deserialize_streamable_list<NodeType,ListType>( src, dest );
			}
		};


                /**
                   list_serializable_proxy is a functor for de/serializing lists
                   of Serializables.
                */
                class list_serializable_proxy : ::s11n::serialize_binary_f_tag,
						::s11n::deserialize_binary_f_tag
                {
                public:
                        list_serializable_proxy()
                        {}

                        /**
                           see serialize_list().

                        */
                        template <typename NodeType, typename SerType>
                        inline bool operator()( NodeType & dest, const SerType & src ) const
                        {
                                return serialize_list( dest, src );
                        }

                        /** see deserialize_list(). */
                        template <typename NodeType, typename SerType>
                        inline bool operator()( const NodeType & src, SerType & dest ) const
                        {
                                return deserialize_list( src, dest );
                        }
                };

                /**
                   streamable_list_serializable_proxy is a functor for de/serializing lists
                   of i/ostreamable Serializables (e.g., PODs).
                */
                class streamable_list_serializable_proxy : ::s11n::serialize_binary_f_tag,
							   ::s11n::deserialize_binary_f_tag
                {
                public:
                        streamable_list_serializable_proxy()
                        {}

                        /**
                           see serialize_streamable_list().

                        */
                        template <typename NodeType, typename SerType>
                        inline bool operator()( NodeType & dest, const SerType & src ) const
                        {
                                return serialize_streamable_list( dest, src );
                        }

                        /** see deserialize_streamable_list(). */
                        template <typename NodeType, typename SerType>
                        inline bool operator()( const NodeType & src, SerType & dest ) const
                        {
                                return deserialize_streamable_list( src, dest );
                        }
                };



        } // namespace list
} // namespace s11n

#include <s11n.net/s11n/proxy/listish.tpp> // implementations

#endif // s11n_net_s11n_v1_1_LIST_HPP_INCLUDED
