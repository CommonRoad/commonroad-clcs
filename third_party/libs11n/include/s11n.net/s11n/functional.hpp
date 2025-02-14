#ifndef s11n_net_s11n_1_1_FUNCTIONAL_HPP_INCLUDED
#define s11n_net_s11n_1_1_FUNCTIONAL_HPP_INCLUDED 1
// Experimental code for s11n, with an emphasis on the "mental" part.
// Here be beg, borrow, and steal many ideas from metatemplate and
// functional libraries, like Boost.MPL.

#include <s11n.net/s11n/tags.hpp>

namespace s11n {

	/**
	   Holds a [const] reference to an object. For use in template
	   metaprogramming. Don't use this type directly: it is intended
	   to be subclassed by reference_f<> types.

	   T must not be pointer-qualified. Const is fine.

	   Added in 1.1.3.

	   Note to self: what happens if it is a pointer type?
	*/
	template <typename T>
	struct reference_base_f
	{
		typedef T type;
		type & value;
		explicit reference_base_f( type & _ref ) : value(_ref)
		{}

		/**
		   Returns a copy of the object this one refers to.
		*/
		inline operator T() const
		{
			return this->value;
		}

		/**
		   Returns a [const] reference to the object this one
		   refers to.
		*/
		inline type & operator()() const
		{
			return this->value;
		}

	};

	/**
	   Holds a reference to an object. For use in template
	   metaprogramming.

	   Added in 1.1.3.
	*/
	template <typename T>
	struct reference_f : reference_base_f<T>
	{
		typedef typename reference_base_f<T>::type type;
		explicit reference_f( type & _ref ) : reference_base_f<T>(_ref)
		{}

		template <typename X>
		inline const reference_f & operator=( const X & val ) const
		{
			this->value = val;
			return *this;
		}
	};

	/**
	   Holds a const reference to an object. For use in template
	   metaprogramming.

	   Added in 1.1.3.
	*/
	template <typename T>
	struct reference_f<T const> : reference_base_f<T const>
	{
		typedef typename reference_base_f<T const>::type type;
		explicit reference_f( type & _ref ) : reference_base_f<T const>(_ref)
		{}
	};

	/**
	   Added in 1.1.3.
	*/
	template <typename T>
	struct reference_f<T *> : reference_base_f<T>
	{
		/** _ref must not be 0. */
		typedef typename reference_base_f<T>::type type;
		explicit reference_f( type * _ref ) : reference_base_f<T>(*_ref)
		{}
	};

	/**
	   Added in 1.1.3.
	*/
	template <typename T>
	struct reference_f<T const *> : reference_base_f<T const>
	{
		/** _ref must not be 0. */
		typedef typename reference_base_f<T const>::type type;
		explicit reference_f( type * _ref ) : reference_base_f<T const>(*_ref)
		{}
	};


	/**
	   Convenience function to return reference_f<T>(ref).

	   Added in 1.1.3.
	*/
	template <typename T>
	inline reference_f<T> ref( T & ref )
	{
		return reference_f<T>(ref);
	}


	/**
	   Pointer equivalent of reference_base_f.

	   Added in 1.1.3.

	   T may be const or not, but should not have a pointer
	   qualification. Specializations of pointer_f take care of
	   getting rid of the extra const/pointer/reference
	   qualifiers.
	*/
	template <typename T>
	struct pointer_base_f
	{
		/** Same as T, possibly const-qualified. */
		typedef T type;
		type * value;
		explicit pointer_base_f( type * _ref ) : value(_ref)
		{}
		explicit pointer_base_f( type & _ref ) : value(&_ref)
		{}
		/**
		   Returns a [const] pointer to the object this one
		   refers to.
		*/
		inline type * operator()() const
		{
			return this->value;
		}

		/** For convention's sake... */
		inline type * get() const
		{
			return this->value;
		}

		inline type * operator->() const
		{
			return this->value;
		}

		inline bool empty() const
		{
			return 0 != this->value;
		}
	};

	/**
	   Pointer equivalent of reference_f.
	   Added in 1.1.3.
	*/
	template <typename T>
	struct pointer_f : pointer_base_f<T>
	{
		typedef typename pointer_base_f<T>::type type;
		explicit pointer_f( type & _ref ) : pointer_base_f<T>(&_ref)
		{}
		explicit pointer_f( type * _ref ) : pointer_base_f<T>(_ref)
		{}
	};

	/**
	   Pointer equivalent of reference_f.
	   Added in 1.1.3.
	*/
	template <typename T>
	struct pointer_f<T const> : pointer_base_f<T const>
	{
		typedef typename pointer_base_f<T const>::type type;
		explicit pointer_f( type & _ref ) : pointer_base_f<T const>(&_ref)
		{}
		explicit pointer_f( type * _ref ) : pointer_base_f<T const>(_ref)
		{}

	};

	/**
	   Pointer equivalent of reference_f.
	   Added in 1.1.3.
	*/
	template <typename T>
	struct pointer_f<T *> : pointer_base_f<T>
	{
		typedef typename pointer_base_f<T>::type type;
		explicit pointer_f( type * _ref ) : pointer_base_f<T>(_ref)
		{}
		explicit pointer_f( type & _ref ) : pointer_base_f<T>(&_ref)
		{}
	};

	/**
	   Pointer equivalent of reference_f.
	   Added in 1.1.3.
	*/
	template <typename T>
	struct pointer_f<T const *> : pointer_base_f<T const>
	{
		typedef typename pointer_base_f<T const>::type type;
		explicit pointer_f( type * _ref ) : pointer_base_f<T const>(_ref)
		{}
		explicit pointer_f( type & _ref ) : pointer_base_f<T const>(&_ref)
		{}
	};

// 	/** Returns pointer_f<T>(v). */
// 	template <typename T>
// 	inline pointer_f<T> pointer( typename type_traits<T>::type & v )
// 	{
// 		return pointer_f<T>(v);
// 	}

//  	template <typename T>
//  	inline pointer_f<T> pointer( typename type_traits<T>::type * v )
//  	{
//  		return pointer_f<T>(v);
//  	}

//  	template <typename T>
//  	inline pointer_f<T const> pointer( typename type_traits<T const>::type const & v )
//  	{
//  		return pointer_f<T const>(v);
//  	}

//  	template <typename T>
//  	inline pointer_f<T const> pointer( typename type_traits<T const>::type const * v )
//  	{
//  		return pointer_f<T const>(v);
//  	}


	/**
	   Holds a value. For use in template metaprogramming.

	   Added in 1.1.3.
	*/
	template <typename T>
	struct value_f
	{
		typedef T type;
		type value;
		value_f( type const & _ref ) : value(_ref)
		{}

		/** Returns a copy of this->value. */
		inline operator T() const
		{
			return this->value;
		}

		/** Returns a copy of this->value. */
		inline type operator()() const
		{
			return this->value;
		}
	};

	/** Quasi-bogus specialization. */
	template <typename T>
	struct value_f<T const> : value_f<T> {};

	/** Quasi-bogus specialization. */
	template <typename T>
	struct value_f<T &> : value_f<T> {};

	/** Quasi-bogus specialization. */
	template <typename T>
	struct value_f<T const &> : value_f<T> {};


	/** Returns value_f<T>(v). */
	template <typename T>
	inline value_f<T> val( const T & v )
	{
		return value_f<T>(v);
	}



	/**
	   A functor which simply forwards its arguments to
	   s11n::serialize().

	   Added in 1.1.3.
	*/
	struct serialize_f : serialize_binary_f_tag
	{
		template <typename NT, typename ST>
		inline bool operator()( NT & dest, const ST & src ) const
		{
			return serialize<NT,ST>( dest, src );
		}
	};

	/**
	   A functor which simply forwards its arguments to
	   s11n::deserialize().

	   Added in 1.1.3.
	*/
	struct deserialize_f : deserialize_binary_f_tag
	{
		template <typename NT, typename ST>
		inline bool operator()( const NT & src, ST & dest ) const
		{
			return deserialize<NT,ST>( src, dest );
		}
	};


	/**
	   Conforms to serialize_nullary_f_tag expectations
	   and converts a serialize_binary_f_tag type to
	   serialize_nullary_f_tag type.

	   BinaryFunctorT must comply to serialize_binary_f_tag's
	   expectations.

	   Under NO circumstances may you pass TEMPORARY as a SerializableT
	   argument to one of the ctors. This is strictly illegal, as
	   we hold a reference to the object.

	   Added in 1.1.3.
	*/
	template <typename NodeType,typename SerializableT, typename BinaryFunctorT = serialize_f>
	struct serialize_nullary_f : serialize_nullary_f_tag
	{
		reference_f<NodeType> node;
		reference_f<SerializableT const> serializable;
		BinaryFunctorT functor;
		serialize_nullary_f( NodeType & n, SerializableT const & s )
			: node(n), serializable(s), functor()
		{
		}

		serialize_nullary_f( NodeType & n, SerializableT const & s, BinaryFunctorT const & f ): node(n), serializable(s), functor(f)
		{
		}
		/** Returns this->functor( this->node, this->serializable ). */
		inline bool operator()() const
		{
			return this->functor( this->node(), this->serializable() );
		}
	};

	/**
	   Returns serialize_nullary_f<NodeType,SerializableT,BinaryFunctorT>( n, s, f ).
	*/
	template <typename NodeType,typename SerializableT, typename BinaryFunctorT>
	inline serialize_nullary_f<NodeType,SerializableT,BinaryFunctorT>
	ser_nullary_f( NodeType & n, SerializableT const & s, BinaryFunctorT const & f )
	{
		return serialize_nullary_f<NodeType,SerializableT,BinaryFunctorT>( n, s, f );
	}

	/**
	   Returns serialize_nullary_f<NodeType,SerializableT>( n, s ).
	*/
	template <typename NodeType,typename SerializableT>
	inline serialize_nullary_f<NodeType,SerializableT>
	ser_nullary_f( NodeType & n, SerializableT const & s )
	{
		return serialize_nullary_f<NodeType,SerializableT>( n, s );
	}

	/**
	   Converts an S-Node to a unary functor taking a Serializable
	   argument.

	   BinaryFunctorT must comply to serialize_binary_f_tag.
	*/
 	template <typename NodeType, typename BinaryFunctorT = serialize_f>
 	struct node_to_serialize_unary_f : serialize_unary_serializable_f_tag
	{
		//typedef NodeType type;
		reference_f<NodeType> node;
		BinaryFunctorT functor;
		node_to_serialize_unary_f( NodeType & n ) : node(n), functor() {}
		node_to_serialize_unary_f( NodeType & n, BinaryFunctorT const & f ) : node(n), functor(f) {}

		template <typename SerT>
		inline bool operator()( const SerT & src ) const
		{
			return this->functor( this->node(), src );
		}
	};

	/**
	   Returns node_to_serialize_unary_f<NodeType,BinaryFunctorT>(n,f).
	*/
 	template <typename NodeType, typename BinaryFunctorT>
 	node_to_serialize_unary_f<NodeType,BinaryFunctorT>
	node_to_ser_unary_f( NodeType & n, BinaryFunctorT f )
	{
		return node_to_serialize_unary_f<NodeType,BinaryFunctorT>(n,f);
	}

	/**
	   Returns node_to_serialize_unary_f<NodeType>(n).
	*/
 	template <typename NodeType>
 	node_to_serialize_unary_f<NodeType>
	node_to_ser_unary_f( NodeType & n )
	{
		return node_to_serialize_unary_f<NodeType>(n);
	}


	/**
	   Converts an S-Node to a unary functor taking a Serializable
	   argument.

	   BinaryFunctorT must comply to deserialize_binary_f_tag.
	*/
 	template <typename NodeType, typename BinaryFunctorT = deserialize_f>
 	struct node_to_deserialize_unary_f : deserialize_unary_serializable_f_tag
	{
		//typedef NodeType type;
		reference_f<NodeType const> node;
		BinaryFunctorT functor;
		node_to_deserialize_unary_f( NodeType const & n ) : node(n), functor() {}
		node_to_deserialize_unary_f( NodeType const & n, BinaryFunctorT const & f ) : node(n), functor(f) {}
		template <typename SerT>
		inline bool operator()( SerT & dest ) const
		{
			return this->functor( this->node(), dest );
		}
	};

	/**
	   Returns node_to_deserialize_unary_f<NodeType,BinaryFunctorT>(n,f).
	*/
 	template <typename NodeType, typename BinaryFunctorT>
 	node_to_deserialize_unary_f<NodeType,BinaryFunctorT>
	node_to_deser_unary_f( NodeType const & n, BinaryFunctorT const & f )
	{
		return node_to_deserialize_unary_f<NodeType,BinaryFunctorT>(n,f);
	}

	/**
	   Returns node_to_deserialize_unary_f<NodeType>(n).
	*/
 	template <typename NodeType>
 	inline node_to_deserialize_unary_f<NodeType>
	node_to_deser_unary_f( NodeType const & n )
	{
		return node_to_deserialize_unary_f<NodeType>(n);
	}


	/**
	   Converts a Serializable to a type compliant with
	   serialize_unary_node_f_tag.

	   BinaryFunctorT must comply to serialize_binary_f_tag.

	   Added in 1.1.3.
	*/
 	template <typename SerT, typename BinaryFunctorT = serialize_f>
 	struct serializable_to_serialize_unary_f : serialize_unary_node_f_tag
	{
		typedef SerT type;
		reference_f<SerT const> serializable;
		BinaryFunctorT functor;
		serializable_to_serialize_unary_f( SerT const & n ) : serializable(n), functor() {}
		serializable_to_serialize_unary_f( SerT const & n, BinaryFunctorT const & f ) : serializable(n), functor(f) {}

		template <typename NodeT>
		inline bool operator()( NodeT & dest ) const
		{
			return this->functor( dest, this->serializable() );
		}
	};

	/**
	   Returns serializable_to_serialize_unary_f<SerT,BinaryFunctorT>( s, f ).
	*/
 	template <typename SerT, typename BinaryFunctorT>
 	inline serializable_to_serialize_unary_f<SerT,BinaryFunctorT>
	ser_to_ser_unary_f( SerT const & s, BinaryFunctorT const & f )
	{
		return serializable_to_serialize_unary_f<SerT,BinaryFunctorT>( s, f );
	}

	/**
	   Returns serializable_to_serialize_unary_f<SerT>( s ).
	*/
 	template <typename SerT>
 	inline serializable_to_serialize_unary_f<SerT>
	ser_to_ser_unary_f( SerT const & s)
	{
		return serializable_to_serialize_unary_f<SerT>( s );
	}
	

	/**
	   Converts a Serializable to a type compliant with
	   deserialize_unary_node_f_tag.

	   BinaryFunctorT must comply to deserialize_binary_f_tag.

	   Added in 1.1.3.
	*/
 	template <typename SerT, typename BinaryFunctorT = deserialize_f>
 	struct serializable_to_deserialize_unary_f : deserialize_unary_node_f_tag
	{
		typedef SerT type;
		reference_f<SerT> serializable;
		BinaryFunctorT functor;
		serializable_to_deserialize_unary_f( SerT & n ) : serializable(n), functor() {}
		serializable_to_deserialize_unary_f( SerT & n, BinaryFunctorT const & f ) : serializable(n), functor(f) {}

		template <typename NodeT>
		inline bool operator()( NodeT const & src ) const
		{
			return this->functor( src, this->serializable() );
		}
	};

	/**
	   Returns serializable_to_deserialize_unary_f<SerT,BinaryFunctorT>( s, f ).
	*/
 	template <typename SerT, typename BinaryFunctorT>
 	inline serializable_to_deserialize_unary_f<SerT,BinaryFunctorT>
	ser_to_deser_unary_f( SerT & s, BinaryFunctorT const & f )
	{
		return serializable_to_deserialize_unary_f<SerT,BinaryFunctorT>( s, f );
	}

	/**
	   Returns serializable_to_deserialize_unary_f<SerT>( s ).
	*/
 	template <typename SerT>
 	inline serializable_to_deserialize_unary_f<SerT>
	ser_to_deser_unary_f( SerT const & s)
	{
		return serializable_to_deserialize_unary_f<SerT>( s );
	}



	/**
	   Conforms to deserialize_nullary_f_tag expectations
	   and converts a deserialize_binary_f_tag type to
	   deserialize_nullary_f_tag type.

	   BinaryFunctorT must comply to deserialize_binary_f_tag's
	   expectations.

	   Under NO circumstances may you pass TEMPORARY as a NodeType
	   argument to one of the ctors. This is strictly illegal, as
	   we hold a reference to the object.

	   Added in 1.1.3.
	*/
	template <typename NodeType,typename DeserializableT, typename BinaryFunctorT = deserialize_f>
	struct deserialize_nullary_f : deserialize_nullary_f_tag
	{
		reference_f<NodeType const> node;
		reference_f<DeserializableT> serializable;
		BinaryFunctorT functor;
		deserialize_nullary_f( NodeType const & n, DeserializableT & s )
			: node(n), serializable(s), functor()
		{
		}

		deserialize_nullary_f( NodeType const & n, DeserializableT & s, BinaryFunctorT const & f )
			: node(n), serializable(s), functor(f)
		{
		}

		/**
		   Returns this->functor( this->node, this->serializable ).
		*/
		inline bool operator()() const
		{
			return this->functor( this->node(), this->serializable() );
		}
	};

	/**
	   Returns deserialize_nullary_f<NodeType,DeserializableT,BinaryFunctorT>( n, s, f ).
	*/
	template <typename NodeType,typename DeserializableT, typename BinaryFunctorT>
	inline deserialize_nullary_f<NodeType,DeserializableT,BinaryFunctorT>
	deser_nullary_f( NodeType const & n, DeserializableT & s, BinaryFunctorT const & f )
	{
		return deserialize_nullary_f<NodeType,DeserializableT,BinaryFunctorT>( n, s, f );
	}

	/**
	   Returns deserialize_nullary_f<NodeType,DeserializableT>( n, s );
	*/
	template <typename NodeType,typename DeserializableT>
	inline deserialize_nullary_f<NodeType,DeserializableT>
	deser_nullary_f( NodeType const & n, DeserializableT & s )
	{
		return deserialize_nullary_f<NodeType,DeserializableT>( n, s );
	}



	/**
	   Experimental. Added in 1.1.3.

	   A Serializable functor intended for some metaprogramming
	   experimentation, to allow lazy s11n of a Serializable.

	   BinaryFunctorT requires this signature:

	   bool operator()( NodeType & dest, const SerializableT & src )
	 */
	template <typename SerializableT, typename BinaryFunctorT = serialize_f>
	struct serializable_f : serialize_unary_node_f_tag
	{
		typedef SerializableT const type;
		reference_f<type> reference;
		BinaryFunctorT functor;

		/**
		   Sets this->reference(_ref).
		 */
		explicit serializable_f( type & _ref ) : reference(_ref), functor()
		{
		}

		serializable_f( type & _ref, BinaryFunctorT f ) : reference(_ref), functor(f)
		{
		}

		/**
		   Returns serialize( dest, this->ref ).

		   Calling after this->ref has been destroyed
		   yields undefined behaviour.
		*/
		template <typename NodeType>
		inline bool operator()( NodeType & dest ) const
		{
			return this->functor( dest, this->reference() );
		}

		/** Retyurns a const reference to this object's referenced Serializable. */
		inline type & operator()() const
		{
			return this->reference();
		}
	};

	/**
	   Experimental. Added in 1.1.3.

	   Returns serializable_f<SerializableT>( ref ).

	   ref must outlive the object returned by this function!
	*/
	template <typename SerializableT>
	inline serializable_f<SerializableT>
	ser_f( SerializableT const & ref )
	{
		return serializable_f<SerializableT>( ref );
	}

	/**
	   Experimental. Added in 1.1.3.

	   Returns serializable_f<SerializableT,BinaryFunctorT>( ref, f ).

	   ref must outlive the object returned by this function!
	*/
	template <typename SerializableT,typename BinaryFunctorT>
	inline serializable_f<SerializableT,BinaryFunctorT>
	ser_f( SerializableT const & ref, BinaryFunctorT f )
	{
		return serializable_f<SerializableT,BinaryFunctorT>( ref, f );
	}

	/**
	   Experimental. Added in 1.1.3.

	   A Serializable functor intended for some metaprogramming
	   experimentation, to allow lazy de-s11n of a Serializable.

	   BinaryFunctorT requires this signature:

	   bool operator()( const NodeType & src, SerializableT & dest )

	*/
	template <typename DeserializableT, typename BinaryFunctorT = deserialize_f>
	struct deserializable_f : deserialize_unary_node_f_tag
	{
		//typedef deserializable_f<DeserializableT,BinaryFunctorT> type;
		typedef DeserializableT type;
		reference_f<type> reference;
		BinaryFunctorT functor;

		/**
		   Sets this->ref = _ref.
		 */
		explicit deserializable_f( type & _ref ) : reference(_ref),functor()
		{
		}

		deserializable_f( type & _ref, BinaryFunctorT f ) : reference(_ref),functor(f)
		{
		}
		/**
		   Returns deserialize( src, this->ref ).

		   Calling after this->ref has been destroyed
		   yields undefined behaviour.
		 */
		template <typename NodeType>
		inline bool operator()( const NodeType & src ) const
		{
			return this->functor( src, this->reference() );
		}

		/** Retyurns a reference to this object's referenced Serializable. */
		inline type & operator()() const
		{
			return this->reference();
		}
	};


	/**
	   Added in 1.1.3.

	   Returns deserializable_f<DeserializableT>( ref ).

	   ref must outlive the object returned by this function!
	*/
	template <typename DeserializableT>
	inline deserializable_f<DeserializableT>
	deser_f( DeserializableT & ref )
	{
		return deserializable_f<DeserializableT>( ref );
	}

 	/**
	   Added in 1.1.3.

	   Returns deserializable_f<DeserializableT,BinaryFunctorT>( ref, f ).

	   ref must outlive the object returned by this function!
	*/
	template <typename DeserializableT,typename BinaryFunctorT>
	inline deserializable_f<DeserializableT,BinaryFunctorT>
	deser_f( DeserializableT & ref, BinaryFunctorT f )
	{
		return deserializable_f<DeserializableT,BinaryFunctorT>( ref, f );
	}

	/**
	   A functor to allow us to easily walk a list of S-Nodes and
	   deserialize each one into a target container.


	   The optional BinaryFunctorT defines the functor to use
	   to deserialize each object. The default simply routes
	   through the s11n::deserialize() API.

	   SerializableType is unforuntately required: we can't
	   derive it from the output iterator.

	   SerializableType MAY NOT yet be pointer-qualified. That's on
	   the to-fix list somewhere. It would inherently cause a leak
	   or be very incorrect in some uses, though, like using an
	   ostream_iterator(). It could be made to function, but would not
	   be leak-proof.

	   Also, we create and copy SerializableTypes here, so that type
	   should be cheap to do that with.

	   BinaryFunctorT must conform to the interface defined by
	   deserialize_f.

	   Example, assuming NTR is a node_traits type:

	   <pre>
	   std::for_each( NTR::children(node).begin(),
		          NTR::children(node).end(),
		          deser_to_outiter_f<MyType>( std::back_inserter(myvector) ) );
	   </pre>
	*/
	template <typename SerializableType, typename OutIterator, typename BinaryFunctorT = deserialize_f>
	struct deserialize_to_output_iter_f
	{
		// typedef deserialize_to_output_iter_f<SerializableType,OutIterator,BinaryFunctorT> type;
		typedef OutIterator type;
		type iterator;
		BinaryFunctorT functor;
		typedef SerializableType serializable_type;
		// typedef typename std::iterator_traits<OutIterator>::value_type serializable_type;
		// ^^^ why is this void?

		/**
		   Sets this object's output iterator.
		*/
		explicit deserialize_to_output_iter_f( type target ) : iterator(target), functor()
		{}

		/**
		   Sets this object's output iterator and copies the
		   given functor.
		*/
		deserialize_to_output_iter_f( type target, BinaryFunctorT f ) : iterator(target), functor(f)
		{}

		/**
		   Creates a new object of serializable_type and deserializes it.
		   On success iterator is assigned and incremented and true is returned.
		   On error false is returned or an exception is propagated.

		   If src is (!src), false is returned.

		   Note that the odd pointerness of the argument is because
		   node children lists contain pointers and are const
		   in a deserialize context.
		*/
		template <typename NodeType>
		bool operator()( NodeType * const & src )
		{
			if( src )
			{
				// todo: use classload() to load based
				// on src's class_name() and wrap that
				// pointer in a cleanup_ptr.
				serializable_type dest;
				if( this->functor( *src, dest ) )
				{
					*(iterator++) = dest;
					return true;
				}
			}
			return false;
		}
	};

	/**
	   Convenience function returning:

	   deserialize_to_output_iter_f<SerializableType,OutIterator,BinaryFunctorT>( target, f )
	*/
	template <typename SerializableType,typename OutIterator, typename BinaryFunctorT>
	inline deserialize_to_output_iter_f<SerializableType,OutIterator,BinaryFunctorT>
	deser_to_outiter_f( OutIterator target, BinaryFunctorT f )
	{
		return deserialize_to_output_iter_f<SerializableType,OutIterator,BinaryFunctorT>( target, f );
	}

	/**
	   Convenience function returning:

	   deserialize_to_output_iter_f<SerializableType,OutIterator>( target )
	*/
	template <typename SerializableType, typename OutIterator>
	inline deserialize_to_output_iter_f<SerializableType,OutIterator>
	deser_to_outiter_f( OutIterator target )
	{
		return deserialize_to_output_iter_f<SerializableType,OutIterator>( target );
	}


 	/**
	   Experimental. Added in 1.1.3.

	   BinaryFunctorT must have:

	   bool operator()( NodeType & dest, const SerializableT & src )

	*/
 	template <typename BinaryFunctorT = serialize_f>
 	struct serialize_to_subnode_f : serialize_binary_f_tag
 	{
		//typedef serialize_to_subnode_f<BinaryFunctorT> type;
 		std::string name;
		BinaryFunctorT functor;


		/**
		*/
 		serialize_to_subnode_f( const std::string & subnodename, BinaryFunctorT f )
 			: name(subnodename),functor(f)
 		{
 		}

		/**
		*/
 		explicit serialize_to_subnode_f( const std::string & subnodename )
 			: name(subnodename),functor()
 		{
 		}

		/**
		   Creates a NodeType names this->name and calls
		   this->functor( child, src ). If the functor fails,
		   the child is deleted and dest is unmodified, else
		   ownership of the child is transfered to dest, via
		   node_traits<NodeType>::children(dest).push_back(child).

		   Returns true on success, false on failure.
		*/
 		template <typename NodeType, typename SerializableT>
 		inline bool operator()( NodeType & dest, SerializableT const & src ) const
 		{
			typedef node_traits<NodeType> NTR;
			std::unique_ptr<NodeType> nap( NTR::create( this->name ) );
 			return this->functor( *nap, src )
				? (NTR::children(dest).push_back( nap.release() ),true)
				: false;
 		}

 	};



 	/**
	   Returns serialize_to_subnode_f<>( subnodename ).

	   Added in 1.1.3.
	*/
 	inline serialize_to_subnode_f<>
 	ser_to_subnode_f( const std::string & subnodename )
 	{
 		return serialize_to_subnode_f<>( subnodename );
 	}

 	/**
	   Returns serialize_to_subnode_f<BinaryFunctorT>( subnodename, f ).

	   Added in 1.1.3.
	*/
 	template <typename BinaryFunctorT>
 	inline serialize_to_subnode_f<BinaryFunctorT>
 	ser_to_subnode_f( const std::string & subnodename, BinaryFunctorT f )
 	{
 		return serialize_to_subnode_f<BinaryFunctorT>( subnodename, f );
 	}

	

 	/**
	   Experimental. Added in 1.1.3.

	   BinaryFunctorT must have:

	   bool operator()( NodeType & dest, const SerializableT & src )

	*/
 	template <typename NodeType, typename BinaryFunctorT = serialize_f>
 	struct serialize_to_subnode_unary_f : serialize_unary_serializable_f_tag
 	{
		//typedef serialize_to_subnode_unary_f<NodeType,BinaryFunctorT> type;
 		reference_f<NodeType> node;
 		std::string name;
		BinaryFunctorT functor;

		/**
		*/
 		serialize_to_subnode_unary_f( NodeType & parent, const std::string & subnodename )
 			: node(parent), name(subnodename),functor()
 		{
 		}

		/**
		*/
 		serialize_to_subnode_unary_f( NodeType & parent, const std::string & subnodename, BinaryFunctorT f )
 			: node(parent), name(subnodename),functor(f)
 		{
 		}

		/**
		   Creates a NodeType named this->name and calls
		   this->functor( child, src ). If the functor fails,
		   the child is deleted and dest is unmodified, else
		   ownership of the child is transfered to dest, via
		   node_traits<NodeType>::children(this->node()).push_back(child).

		   Returns true on success, false on failure.
		   
		*/
 		template <typename SerializableT>
 		bool operator()( SerializableT const & src ) const
 		{
			typedef node_traits<NodeType> NTR;
			std::unique_ptr<NodeType> nap( NTR::create( this->name ) );
 			if( this->functor( *nap, src ) )
			{
				NTR::children(this->node()).push_back( nap.release() );
			}
			return 0 == nap.get();
 		}
 	};


 	/**
	   Returns serialize_to_subnode_f<NodeType>( parent, subnodename ).

	   Example:
	   <pre>
	   std::for_each( vec.begin(),
		          vec.end(),
		          ser_to_subnode_unary_f( mynode, "child" )
		        );
	   </pre>

	   Added in 1.1.3.
	*/
 	template <typename NodeType>
 	inline serialize_to_subnode_unary_f<NodeType>
 	ser_to_subnode_unary_f( NodeType & parent, const std::string & subnodename )
 	{
 		return serialize_to_subnode_unary_f<NodeType>( parent, subnodename );
 	}

 	/**
	   Returns serialize_to_subnode_unary_f<NodeType,BinaryFunctorT>( parent, subnodename, f ).

	   Example:

	   <pre>
	   std::for_each( vec.begin(),
		       vec.end(),
		       ser_to_subnode_f( mynode, "child", my_serialize_to_subnode_unary_functor() )
		       );
	   </pre>

	   Added in 1.1.3.
	*/
 	template <typename NodeType, typename BinaryFunctorT>
 	inline serialize_to_subnode_unary_f<NodeType,BinaryFunctorT>
 	ser_to_subnode_unary_f( NodeType & parent, const std::string & subnodename, BinaryFunctorT f )
 	{
 		return serialize_to_subnode_unary_f<NodeType,BinaryFunctorT>( parent, subnodename, f );
 	}



 	/**
	   

	   BinaryFunctorT requires this signature:

	   bool operator()( const NodeType & src, const std::string & subnodename, SerializableT & dest )

	   Added in 1.1.3.
	*/
 	template <typename BinaryFunctorT = deserialize_f>
 	struct deserialize_from_subnode_f : deserialize_binary_f_tag
 	{
		//typedef deserialize_from_subnode_f<BinaryFunctorT> type;
 		std::string name;
		BinaryFunctorT functor;
 		deserialize_from_subnode_f( const std::string & subnodename )
 			: name(subnodename),functor()
 		{
 		}

 		deserialize_from_subnode_f( const std::string & subnodename, BinaryFunctorT f )
 			: name(subnodename),functor(f)
 		{
 		}

		/**
		   Searches for the first child in src named this->name. If it succeeds,
		   it returns the result of this->functor( child, dest ), else it returns
		   false.
		*/
 		template <typename NodeType, typename SerializableT>
 		inline bool operator()( NodeType const & src, SerializableT & dest ) const
 		{
			const NodeType * ch = ::s11n::find_child_by_name( src, this->name );
			return ch
				? this->functor( *ch, dest )
				: false;
 		}
 	};


 	/**
	   Returns deserialize_from_subnode_f<>( parent, subnodename ).

	   Added in 1.1.3.
	*/
 	inline deserialize_from_subnode_f<>
 	deser_from_subnode_f( const std::string & subnodename )
 	{
 		return deserialize_from_subnode_f<>( subnodename );
 	}
 	/**
	   Returns deserialize_from_subnode_f<BinaryFunctorT>( parent, subnodename, f ).

	   Added in 1.1.3.
	*/
 	template <typename BinaryFunctorT>
 	inline deserialize_from_subnode_f<BinaryFunctorT>
 	deser_from_subnode_f( const std::string & subnodename, BinaryFunctorT f )
 	{
 		return deserialize_from_subnode_f<BinaryFunctorT>( subnodename, f );
 	}

 	/**
	   BinaryFunctorT requires this signature:

	   bool operator()( const NodeType & src, const std::string & subnodename, SerializableT & dest )

	   Added in 1.1.3.
	*/
 	template <typename NodeType, typename BinaryFunctorT = deserialize_f>
 	struct deserialize_from_subnode_unary_f : deserialize_unary_serializable_f_tag
 	{
		//typedef deserialize_from_subnode_unary_f<NodeType,BinaryFunctorT> type;
 		reference_f<NodeType const> node;
 		std::string name;
		BinaryFunctorT functor;
 		deserialize_from_subnode_unary_f( const NodeType & parent, const std::string & subnodename )
 			: node(parent), name(subnodename),functor()
 		{
 		}

 		deserialize_from_subnode_unary_f( const NodeType & parent, const std::string & subnodename, BinaryFunctorT f )
 			: node(parent), name(subnodename),functor(f)
 		{
 		}

		/**
		   Searches for the first child in this->node() named this->name. If it succeeds,
		   it returns the result of this->functor( child, dest ), else it returns
		   false.
		*/
 		template <typename SerializableT>
 		inline bool operator()( SerializableT & dest ) const
 		{
			const NodeType * ch = ::s11n::find_child_by_name( this->node(), this->name );
			return ch
				? this->functor( *ch, dest )
				: false;
 		}
 	};


 	/**
	   Returns deserialize_from_subnode_unary_f<NodeType>( parent, subnodename ).

	   Added in 1.1.3.
	*/
 	template <typename NodeType>
 	inline deserialize_from_subnode_unary_f<NodeType>
 	deser_from_subnode_unary_f( const NodeType & parent, const std::string & subnodename )
 	{
 		return deserialize_from_subnode_unary_f<NodeType>( parent, subnodename );
 	}
 	/**
	   Experimental. Added in 1.1.3.

	   Returns deserialize_from_subnode_unary_f<NodeType,BinaryFunctorT>( parent, subnodename, f ).
	*/
 	template <typename NodeType, typename BinaryFunctorT>
 	inline deserialize_from_subnode_unary_f<NodeType,BinaryFunctorT>
 	deser_from_subnode_unary_f( const NodeType & parent, const std::string & subnodename, BinaryFunctorT f )
 	{
 		return deserialize_from_subnode_unary_f<NodeType,BinaryFunctorT>( parent, subnodename, f );
 	}




	/**
	   Functor implementing AND logic and between two functors,
	   which are expected to return values convertable to bool.
	*/
	struct logical_and_binary_f
	{
		template <typename F1, typename F2>
		inline bool operator()( F1 const & f1, F2 const & f2 ) const
		{
			return f1() && f2();
		}
	};

	/**
	   Functor implementing AND logic and between two functors,
	   which are expected to return values convertable to bool.
	*/
	template <typename F1>
	struct logical_and_unary_f
	{
		F1 functor;
		logical_and_unary_f( F1 const & f ) : functor(f)
		{}
		template <typename F2>
		inline bool operator()( F2 const & f2 ) const
		{
			return this->functor() && f2();
		}
	};

	/**
	   Functor implementing AND logic and between two functors,
	   which are expected to return values convertable to bool.
	*/
	template <typename F1, typename F2>
	struct logical_and_nullary_f
	{
		F1 functor1;
		F2 functor2;
		logical_and_nullary_f( F1 const & f1, F2 const & f2 ) : functor1(f1), functor2(f2)
		{}
		inline bool operator()() const
		{
			return this->functor1() && this->functor2();
		}
	};


	/**
	   Returns logical_and_nullary<F1,F2>(f1,f2).
	*/
	template <typename F1, typename F2>
	inline logical_and_nullary_f<F1,F2>
	logical_and( F1 const & f1, F2 const & f2 )
	{
		return logical_and_nullary_f<F1,F2>(f1,f2);
	}

	/**
	   Returns logical_and_unary<F1>(f1).
	*/
	template <typename F1>
	inline logical_and_unary_f<F1>
	logical_and( F1 const & f1 )
	{
		return logical_and_unary_f<F1>(f1);
	}

	/**
	   Returns logical_and_binary().
	*/
	inline logical_and_binary_f
	logical_and()
	{
		return logical_and_binary_f();
	}




} // namespace s11n


#endif // s11n_net_s11n_1_1_FUNCTIONAL_HPP_INCLUDED
