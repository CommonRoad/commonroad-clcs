#ifndef s11n_net_s11n_v1_1_MAP_HPP_INCLUDED
#define s11n_net_s11n_v1_1_MAP_HPP_INCLUDED 1
////////////////////////////////////////////////////////////////////////
// Declarations for s11n algos for dealing with std::map and std::pair
// types, or compatible.
////////////////////////////////////////////////////////////////////////

#include <map>
#include <list>
#include <utility> // pair

#include <iterator> // insert_iterator
#include <algorithm> // for_each()

namespace s11n {



        /**
           The s11n::map namespace defines functors and algorithms for
           working with std::map, std::pair, and
           conventions-compatible containers.
        */
        namespace map {


                
                /**

                serialize_streamable_map() is intended for use as a
                serialization proxy for the s11n framework. It
                de/serializes objects of type <tt>std::map&lt;X,Y&gt;</tt>
                into a destination node.

                NodeType must be node_traits<NodeType>-compliant.

                MapType must conform to std::map conventions and
                it's key_type and mapped_type must both be Value Types
                which are i/ostreamable (this includes all PODs and
                std::string). Pointers as keys or values are not
                supported by this functor.

		Unlike serialize_map(), this function cannot work on
		std::multimaps because it uses NodeType's properties
		to hold key/value pairs. The NodeType conventions
		specify unique keys for properties, making them
		incompatible for generic use with multimap containers
		and this algorithm. serialize_map() stores each stored
		item as its own object, where duplicate names are
		allowed, so it is not affected by this limitation.

                Always returns true.

                ACHTUNG: return type changed in 0.9.12.
                */

                template <typename NodeType, typename MapType>
                bool serialize_streamable_map( NodeType & dest, const MapType & src );

                /**
                   Exactly like serialize_streamable_map(dest,src) except that a subnode,
                   named subnodename, of dest is created to store the data. 

                   ACHTUNG: return type changed in 0.9.12.

		   As of version 1.1.3, this function guarantees that dest
		   is not modified if the serialization fails.
                */
                template <typename NodeType, typename MapType>
                bool serialize_streamable_map( NodeType & dest, const std::string & subnodename, const MapType & src );

                struct serialize_streamable_map_f : ::s11n::serialize_binary_f_tag, ::s11n::deserialize_binary_f_tag
		{
			template <typename NodeType, typename MapType>
			bool operator()( NodeType & dest, const MapType & src ) const
			{
				return serialize_streamable_map<NodeType,MapType>( dest, src );
			}
			template <typename NodeType, typename MapType>
			bool operator()( NodeType & dest, const std::string & subnodename, const MapType & src ) const
			{
				return serialize_streamable_map<NodeType,MapType>( dest, subnodename, src );
			}
		};
                /**
                   This is the converse of serialize_streamable_map(). It tries to
                   read in all properties stored in src and stick them into
                   dest.

                   NodeType must support begin() and end() and they must
                   return iterators to pair&lt;X,Y&gt;, where X and Y must
                   meet the same requirements as the key and value types for
                   MapType in serialize_streamable_map(). MapType must support:

                   void insert( MapType::value_type );

                   (Duh.)

                   Always returns true, because the nature of
                   stream-based conversion makes it quite difficult to
                   generically know if there is an error. In practice,
                   such operations essentially always succeed, at
                   least for POD-like types.

                   ACHTUNG: return type changed in 0.9.12.
                   
                */
                template <typename NodeType, typename MapType>
                bool deserialize_streamable_map( const NodeType & src, MapType & dest );


                /**
                   Exactly like deserialize_streamable_map(dest,src) except
                   that a subnode of dest, named subnodename, is sought to
                   pull the data from.
                */
                template <typename NodeType, typename MapType>
                bool deserialize_streamable_map( const NodeType & src, const std::string & subnodename, MapType & dest );

                struct deserialize_streamable_map_f : ::s11n::serialize_binary_f_tag, ::s11n::deserialize_binary_f_tag
		{
			template <typename NodeType, typename MapType>
			bool operator()( const NodeType & src, MapType & dest ) const
			{
				return deserialize_streamable_map<NodeType,MapType>( src, dest );
			}
			template <typename NodeType, typename MapType>
			bool operator()( const NodeType & src, const std::string & subnodename, MapType & dest ) const
			{
				return deserialize_streamable_map<NodeType,MapType>( src, subnodename, dest );
			}
		};

                /**
                   Serializes a std::pair-compatible type into a "custom"
                   format, suitable for saving pairs in standard XML
                   (de/serialize_streamable_map() can't do this when keys are
                   not valid XML keys, e.g., numeric). Use
                   deserialize_streamable_pair() to decode the data.

                   The destination node gets these two properties:

                   - first = src.first

                   - second = src.second

                   PairType must comply with:

                   - first/second types must be i/o streamable (i.e.,
                   convertable to strings).

                   Returns true on success... and never fails. Honestly. It'll
                   fail at compile-time if it's going to fail.

                   use deserialize_streamable_pair() to convert them back to pairs,
                   or fish out the "first" and "second" properties manually.
                */
                template <typename NodeType, typename PairType>
                bool serialize_streamable_pair( NodeType & dest, const PairType & src );

		/**
		   A functor equivalent of serialize_streamable_pair().

		   Added in version 1.1.3.
		 */
                struct serialize_streamable_pair_f : ::s11n::serialize_binary_f_tag
		{
			template <typename NodeType, typename PairType>
			inline bool operator()( NodeType & dest, const PairType & src ) const
			{
				return serialize_streamable_pair<NodeType,PairType>( dest, src );
			}
		};

                /**
                   The quasi-counterpart of serialize_streamable_pair(). It's
                   non-conventional args and return type are a result of
                   map::value_type having a const .first element, which
                   prohibits us assigning to it. See deserialize_pair() for
                   more info on that.

		   As of version 1.1.3, this function throws an
		   s11n_exception if src does not have two properties
		   named 'first' and 'second'.
                */
                template <typename PairType, typename NodeType>
                PairType deserialize_streamable_pair( const NodeType & src  );


		/**
		   Overloaded form with conventional argument ordering.
		   It is NOT suitable for use with PairTypes from map-like containers,
		   as those have const keys.

		   This function throws an s11n_exception if src does
		   not have two properties named 'first' and 'second',
		   in which case dest is not modified.

		   Added in 1.1.3.
		*/
                template <typename NodeType, typename PairType>
                bool deserialize_streamable_pair( const NodeType & src, PairType & dest );

		/**
		   A functor equivalent of deserialize_streamable_pair().

		   Added in version 1.1.3.
		 */
                struct deserialize_streamable_pair_f : ::s11n::serialize_binary_f_tag
		{
			template <typename NodeType, typename PairType>
			inline bool operator()( const NodeType & src, PairType & dest ) const
			{
				return deserialize_streamable_pair<NodeType,PairType>( src, dest );
			}
		};
		
                /**
                   Similar to serialize_streamable_map(), but puts
                   each key/value pair into it's own node, using
                   serialize_streamable_pair(). The end effect is that
                   it's output is more verbose, but may be compatible
                   with more file formats, regardless of the actual
                   key type. e.g., numeric keys are not supported by
                   standard XML (though they are by the s11n XML
                   parsers), and this algorithm structures the data
                   such that this is not a problem.

                   Returns the number of pairs stored.

                   MapType must meet these conditions:

                   value_type must be a pair containing i/ostreamable types
                   (e.g. PODs/strings) with no pointer qualifiers.

                   ACHTUNG: return type changed in 0.9.12.

                */

                template <typename NodeType, typename MapType>
                bool serialize_streamable_map_pairs( NodeType & dest, const MapType & src );

		/**
		   A functor equivalent of serialize_streamable_map_pairs().

		   Added in version 1.1.3.
		 */
                struct serialize_streamable_map_pairs_f : ::s11n::serialize_binary_f_tag
		{
			template <typename NodeType, typename MapType>
			inline bool operator()( NodeType & dest, const MapType & src ) const
			{
				return serialize_streamable_map_pairs<NodeType,MapType>( dest, src );
			}
		};
		
                /**
                   The load-time counterpart to serialize_streamable_map_pairs().

                   ACHTUNG: return type changed in 0.9.12.

                   If it returns false, a child failed to
                   deserialize. In this case, dest is not modified
                   (version 1.1.3+) or in an undefined state (versions
                   older than 1.1.3). On success, dest.swap() is used
                   to populate the map, so dest will lose any contents
                   it had before calling this function. There is one
                   exception to this: if src contains no content for
                   this algorithm, true is returned without modifying
                   dest. In practice, dest is always empty when passed
                   to this function, so this small difference should
                   matter little.

		   An empty src is not an error.

		   In versions prior to 1.1.3, this function was
		   documented as returning true on empty content, but
		   actually (erroneously) returned false. This would show up
		   when de/serializing empty maps.

                */
                template <typename NodeType, typename MapType>
                bool deserialize_streamable_map_pairs( const NodeType & src, MapType & dest );

		/**
		   A functor equivalent of deserialize_streamable_map_pairs().

		   Added in version 1.1.3.
		 */
                struct deserialize_streamable_map_pairs_f : ::s11n::serialize_binary_f_tag
		{
			template <typename NodeType, typename MapType>
			inline bool operator()( const NodeType & src, MapType & dest ) const
			{
				return deserialize_streamable_map_pairs<NodeType,MapType>( src, dest );
			}
		};

                /**
                   serialize_pair() can serialize any std::pair type which
                   meets these conditions:

                   - PairType's first_type and second_type types must both be
                   Serializables. They may be pointer or value types.

		   - PairType may not be a pointer type.

		   If serialization of one child fails, the whole
		   process fails and neither child is added to the
		   dest node. The error, possibly an exception, is
		   propagated back to the caller.

                   ACHTUNG: never pass the same destination container
                   more than once or you will get duplicate and/or
                   incorrect data.

                */
                template <typename NodeType, typename PairType>
                bool serialize_pair( NodeType & dest, const PairType & src );

		/**
		   A functor equivalent of serialize_pair().

		   Added in version 1.1.3.
		 */
                struct serialize_pair_f : ::s11n::serialize_binary_f_tag
		{
			template <typename NodeType, typename PairType>
			inline bool operator()( NodeType & dest, const PairType & src ) const
			{
				return serialize_pair<NodeType,PairType>( dest, src );
			}
		};

                /**
                   The counterpart to serialize_pair().

                   Note: std::map&lt;X,Y&gt;::value_type is not the
                   same as pair&lt;X,Y&gt;, but is pair&lt;const
                   X,Y&gt;, so you cannot simply iterate over a map
                   and pass each pair to this function, because this
                   function cannot assign to the first element of such
                   a pair.

		   Exceptions: this function propagates exceptions if
		   an underlying call to deserialize() throws. If it
		   throws then dest is not modified. Read on for more
		   gory detail...

		   If the assignment operations for
		   PairType::first_type or PairType::second_type throw
		   then dest may be modified if this function throws.

		   Versions prior to 1.1.3 had a destroy-on-error
		   policy, such that objects this function allocates
		   were deleted on error. For this to work as
		   expected, without memory leaks, algorithms which
		   this function calls must clean up in the face of
		   exceptions.  As it turns out, this is a tricky
		   situation, because deserialization can allocate
		   arbitrary objects which might or might not need
		   special clean-up. For example, deserializing a
		   pair<int,list<vector<double*>>*> will cause doubles
		   and lists to be allocated on the heap, and if this function
		   deletes a list containing the vectors which contain
		   those doubles, those pointers leak.

		   Thus...

		   As of 1.1.3, s11n_traits<PairType>::cleanup_functor
		   is used to do cleanup when an an error occurs. This
		   allows us to keep dest unmodified in the face on an
		   error.  For this to work properly, the two
		   s11n_traits::cleanup_functors for
		   PairType::first_type and PairType::second_type must
		   implement proper behaviour.
                */
                template <typename NodeType, typename PairType>
                bool deserialize_pair( const NodeType & src, PairType & dest );

		/**
		   A functor equivalent of deserialize_pair().

		   Added in version 1.1.3.
		 */
                struct deserialize_pair_f : ::s11n::deserialize_binary_f_tag
		{
			template <typename NodeType, typename PairType>
			inline bool operator()( const NodeType & src, PairType & dest ) const
			{
				return deserialize_pair<NodeType,PairType>( src, dest );
			}
		};
        

                /**
                   Serialize the given map into dest. MapType's pairs must be
                   Serializable and must contain Serializable types, but their
                   "pointerness" is irrelevant.

                   See deserialize_map() for important info.

		   If serialization of a child fails, the child is not
		   added to dest and false is returned or any
		   exception is propagated back to the caller. The
		   dest list might be in a partially-populated state,
		   in either case.

		   If this function throws, dest may be in a
		   partially-filled state.

                   ACHTUNG: never pass the same destination container
                   to this function more than once or you will get
                   duplicate and/or incorrect data.

		   As of version 1.1.3, this function throws an s11n_exception
		   if dest is not empty. The reason for this is to enforce
		   that clients do not accidentally re-use the same (populated)
		   node for serialization of multiple objects, which would cause
		   deserialization of the container to fail.
                */
                template <typename NodeType, typename MapType>
                bool serialize_map( NodeType & dest, const MapType & src );



                /**
                   Identical to the two-argument form, but creates a
                   subnode of dest, named subnodename, and serializes
                   to that node.

		   If serialization into the new child node fails, the
		   child node is not added to dest and the error
		   (possibly an exception) is propagated back to the
		   caller.
                */

                template <typename NodeType, typename MapType>
                bool serialize_map( NodeType & dest, const std::string & subnodename, const MapType & src );

		/**
		   A functor equivalent of serialize_map().

		   Added in version 1.1.3.
		 */
                struct serialize_map_f : ::s11n::serialize_binary_f_tag
		{
			template <typename NodeType, typename MapType>
			inline bool operator()( NodeType & dest, const MapType & src ) const
			{
				return serialize_map<NodeType,MapType>( dest, src );
			}

			template <typename NodeType, typename MapType>
			inline bool operator()( NodeType & dest, const std::string & subnodename, const MapType & src ) const
			{
				return serialize_map<NodeType,MapType>( dest, subnodename, src );
			}
		};

                /**
                   The counterpart of serializer_map(), deserializes src into the
                   given map. MapType must be Serializable and contain pairs
                   which themselves are Serializables... ad inifinitum..

		   As of 1.1.3, this function relies on s11n_traits::cleanup_functor
		   for proper cleanup in the face of failed deserialization.
		   See deserialize_pair() for details. Assuming that cleanup
		   routines are in place, this function can now guaranty that
		   dest is not modified if deserialization fails. However,
		   dest must be empty when it is passed to this function,
		   or on success its contents may very well go missing.

                   Minor caveat:

                   This operation will only work with maps containing std::pair
                   types, not map-like classes which use a different pair
                   type. :( The reason is that map&lt;X,Y&gt;::value_type is
                   not a pair of (X,Y), but (const Y,Y), which means we cannot
                   use the map's value_type for a deser operation because we
                   cannot assign to its .first element (i.e., can't
                   deserialize it). To get around that we "manually" create a
                   new std::pair type using map's key_type and mapped_type
                   typedefs, which "loses" the constness for us so we can
                   assign to the first_type during deserialization, and then
                   insert that pair into the deserializing map.

		   MapType must:

		   - not be a pointer type.

		   - have a swap() member function, to support the
		   no-change-on-error feature.
                */
                template <typename NodeType, typename MapType>
                bool deserialize_map( const NodeType & src, MapType & dest );


                /**
                   Identical to the two-argument form, but tries to
                   deserialize from a subnode of src named
                   subnodename. If no such node is found then false is
                   returned, otherwise the result of
                   deserialize_map(thechild,dest) is returned.
                */
                template <typename NodeType, typename MapType>
                bool deserialize_map( const NodeType & src, const std::string & subnodename, MapType & dest );

		/**
		   A functor equivalent of deserialize_map().

		   Added in version 1.1.3.
		 */
                struct deserialize_map_f : ::s11n::deserialize_binary_f_tag
		{
			template <typename NodeType, typename MapType>
			inline bool operator()( const NodeType & src, MapType & dest ) const
			{
				return deserialize_map<NodeType,MapType>( src, dest );
			}

			template <typename NodeType, typename MapType>
			inline bool operator()( const NodeType & src, const std::string & subnodename, MapType & dest ) const
			{
				return deserialize_map<NodeType,MapType>( src, subnodename, dest );
			}
		};


 		/**
		   A proxy for de/serializing pair types holding two
		   i/ostreamable components. It is NOT usable with
 		   pairs from map-like containers because those keys are
		   const, and thus we cannot assign to them.

		   Added in 1.1.3.
		*/
                struct streamable_pair_serializable_proxy : ::s11n::serialize_binary_f_tag, ::s11n::deserialize_binary_f_tag
                {
                        /**
                           Serializes src to dest.

                           ACHTUNG: never pass the same destination container
                           to this operator more than once or you will get
                           duplicate and/or incorrect data.

			   SerializableType must be a std::pair<X,Y>,
			   or compatible. X may not be a contst type.

			   See serialize_streamable_pair() for more details.

                        */
                        template <typename NodeType, typename SerializableType>
                        inline bool operator()( NodeType & dest, const SerializableType & src ) const
                        {
                                return serialize_streamable_pair( dest, src );
                        }

                        /**
                           Deserializes dest from src. See the serialize operator
			   for requirements placed on SerializableType.

			   See deserialize_streamable_pair() for more details.
                        */
                        template <typename NodeType, typename SerializableType>
                        inline bool operator()( const NodeType & src, SerializableType & dest ) const
                        {
                                return deserialize_streamable_pair( src, dest );
                        }
                 };




                /**
                   pair_serializable_proxy is a Serializable Proxy for
                   std::pairs.
                */
                struct pair_serializable_proxy : ::s11n::serialize_binary_f_tag,
						 ::s11n::deserialize_binary_f_tag
                {
                        /**
                           See ::s11n::map::serialize_pair().
                        */
                        template <typename NodeType, typename PairType>
                        inline bool operator()( NodeType & dest, const PairType & src ) const
                        {
                                return serialize_pair( dest, src );
                        }
                        /**
                           See ::s11n::map::deserialize_pair().
                        */
                        template <typename NodeType, typename PairType>
                        inline bool operator()( const NodeType & src, PairType & dest ) const
                        {
                                return deserialize_pair( src, dest );
                        }
                };



                /**
                   A proxy which can serialize std::maps which contain Streamable
                   Types.

                   It uses de/serialize_streamable_map(), so see those
                   functions for details.
                */
                struct streamable_map_serializable_proxy : ::s11n::serialize_binary_f_tag,
							   ::s11n::deserialize_binary_f_tag
                {
                        /**
                           Serializes src to dest.

                           ACHTUNG: never pass the same destination container
                           to this operator more than once or you will get
                           duplicate and/or incorrect data.
                        */
                        template <typename NodeType, typename SerializableType>
                        inline bool operator()( NodeType & dest , const SerializableType & src ) const
                        {
                                return serialize_streamable_map( dest, src );
                        }

                        /**
                           Deserializes dest from src.
                        */
                        template <typename NodeType, typename SerializableType>
                        inline bool operator()( const NodeType & src , SerializableType & dest ) const
                        {
                                return deserialize_streamable_map( src, dest );
                        }
                };



                /**
                   map_serializable_proxy is a Serialization Proxy for std::maps.

                   See de/serialize_map(): this functor simply wraps those.

                */
                struct map_serializable_proxy : ::s11n::serialize_binary_f_tag,
						::s11n::deserialize_binary_f_tag
                {

                        /**
                           Serializes src into dest. Returns true on success,
                           false on error. Uses serialize_map(), so see that
                           function for details.
                        */
                        template <typename NodeType, typename MapType>
                        inline bool operator()( NodeType & dest , const MapType & src ) const
                        {
                                return serialize_map( dest, src );
                        }
                        /**
                           Deserializes src into dest. Returns true on
                           success, false on error. Uses serialize_map(), so
                           see that function for details.
                        */
                        template <typename NodeType, typename MapType>
                        inline bool operator()( const NodeType & src , MapType & dest ) const
                        {
                                return deserialize_map( src, dest );
                        }
                };

		
        } // namespace map
} // namespace s11n

#include <s11n.net/s11n/proxy/mapish.tpp> // implementations for template code
#endif // s11n_net_s11n_v1_1_MAP_HPP_INCLUDED
