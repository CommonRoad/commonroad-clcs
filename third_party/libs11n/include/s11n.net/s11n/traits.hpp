#ifndef s11n_S11N_TRAITS_HPP_INCLUDED
#define s11n_S11N_TRAITS_HPP_INCLUDED 1


#include <vector>
#include <map>

#include <s11n.net/s11n/classload.hpp> // default classloader/factory implementation.
#include <s11n.net/s11n/export.hpp> // S11N_EXPORT_API
#include <s11n.net/s11n/type_traits.hpp>
#include <s11n.net/s11n/s11n_debuggering_macros.hpp>


namespace s11n {

    /**
       node_traits encapsulates information relevant to Data
       Nodes, much in the same way that std::char_traits
       encapsulates character type information.
       
       The default implementation works with
       s11n::s11n_node or API-compatible
       types. Specializations may be defined to work with
       other node types.
       
       By using node_traits, instead of directly accessing a
       Node's API, client code may remain blissfully ignorant of
       the underlying node type.

       All API docs for this class which do not explicitely say
       "this implementation" (or similar) apply to all
       specializations of this type. They act as the "requirements
       document" for implementors of specializations.

       Changes from 1.0.x to 1.1.x:

       - Removed begin() and end(), because they are just as
       easily accessed via children().begin/end(), and it was not
       easy to remember if they returned iterators to the children
       or the properties.

       - Removed iterator typedefs, as they are (almost) as easily
       accessed via the appropriate container's typedefs. Again, this
       was to avoid confusion between the properties and children
       iterator types.
    */
    template <typename NodeT>
    struct S11N_EXPORT_API node_traits
    {
    public:
	/**
	   The same as NodeT.
	*/
	typedef NodeT node_type;

	/**
	   The type uses to store properties for node_type
	   objects.
	*/
	typedef typename node_type::map_type property_map_type;


	/**
	   The type used to store children of node_type
	   objects.
	*/
	typedef typename node_type::child_list_type child_list_type;




	/**
	   Returns a new node_type. The caller owns the
	   returned pointer.

	   It is illegal for this function to return 0. If it
	   cannot create a node for some reason, it must throw
	   an exception.
	*/
	static node_type * create()
	{
	    return new node_type;
	}

	/**
	   Returns a new node_type with the given name. The
	   caller owns the returned pointer.

	   See create() for the no-null-return rule.
	*/
	static node_type * create( const std::string & nodename )
	{
	    node_type * n = create();
	    name( *n, nodename );
	    return n;
	}


	/**
	   Sets the property key to the given value in
	   the given node.

	   ValueT must support complementary ostream<< and
	   istream>> operators.
	*/
	template <typename ValueT>
	static void set( node_type & node,
			 const std::string & key,
			 const ValueT & value )
	{
	    node.set( key, value );
	}

	/**
	   Unsets (removes) the given property from node. It
	   is not an error to unset an non-existing key.
	*/
	static void unset( node_type & node,
			   const std::string & key )
	{
	    node.unset( key );
	}

	/**
	   Returns true if node contains a property
	   named key, else returns false.
	*/
	static bool is_set( const node_type & node,
			    const std::string & key )
	{
	    return node.is_set( key );
	}


	/**
	   Returns an immutable reference to the node's map of properties.
	*/
	static const property_map_type & properties( const node_type & node )
	{
	    return node.properties();
	}

	/**
	   Returns a mutable reference to the node's map of properties.
	*/
	static property_map_type & properties( node_type & node )
	{
	    return node.properties();
	}

	/**
	   Returns the value of the property with the given
	   key, or default_value if that property does not
	   exist or cannot be lexically cast to type ValueT.

	   ValueT must support complementary ostream<< and
	   istream>> operators.
	*/
	template <typename ValueT>
	static ValueT
	get( const node_type & node,
	     const std::string & key,
	     const ValueT & default_value )
	{
	    return node.template get<ValueT>( key, default_value );
	}

	/**
	   Returns a mutable list of children
	   belonging to node.
	*/
	static child_list_type & children( node_type & node )
	{
	    return node.children();
	}

	/**
	   Returns an immutable list of children
	   belonging to node.
	*/
	static const child_list_type & children( const node_type & node )
	{
	    return node.children();
	}

	/**
	   Sets the class name of the type for which node
	   holds serialized data.
	*/
	static void class_name( node_type & node, const std::string & classname )
	{
	    node.class_name( classname );
	}


	/**
	   Returns the class name of the type for which node
	   holds serialized data.
	*/
	static std::string class_name( const node_type & node )
	{
	    return node.class_name();
	}

	/**
	   Sets node's name. See the s11n lib manual for what
	   conventions to follow. In short: the "variable name"
	   rules from most programming languages are a good
	   guideline.
	*/
	static void name( node_type & node, const std::string & name )
	{
	    node.name( name );
	}


	/**
	   Returns node's name.
	*/
	static std::string name( const node_type & node )
	{
	    return node.name();
	}

	/**
	   Removes all children and properties from node,
	   freeing up their resources. Whether the node's
	   name() and class_name() are cleared is
	   implementation-defined. In practice, those are
	   overwritten by algos as needed, so it has not been
	   a concern.
	*/
	static void clear( node_type & node )
	{
	    node.clear();
	}

	/**
	   Returns true if this object has no properties
	   and no children. The name() and class_name()
	   are *not* considered.

	   Added in version 1.1.3.
	*/
	static bool empty( const node_type & node )
	{
	    return node.empty();
	}

	/**
	   Swaps all publically-visible internal state of lhs
	   with that of rhs. This includes:

	   - class_name()

	   - name()

	   - children()

	   - properties()

	   Added in version 1.1.3.
	*/
	static void swap( node_type & lhs, node_type & rhs )
	{
	    return lhs.swap( rhs );
	}

    }; // end node_traits<>


    // 	/**
    // 	   An unfortunate necessity.
    // 	*/
    //         template <typename NodeT>
    //         struct S11N_EXPORT_API node_traits<NodeT const> : node_traits<NodeT> {};


    /**
       A default serialization proxy, which simply
       forwards de/serialize calls to an interface
       implemented as two overloaded member functions
       SerializableType::operator()( NodeT ).
    */
    struct S11N_EXPORT_API default_serialize_functor
    {
	/**
	   Serialize src to dest using src.operator()( dest ).

	   The serialize operator must look like:

	   bool operator()( NodeT & ) const;

	   It may be virtual or a function template.
	*/
	template <typename NodeT, typename SerializableType>
	bool operator()( NodeT & dest, const SerializableType & src ) const
	{
	    return src.operator()( dest );
	}

	/**
	   Deserialize dest from src using dest.operator()( src ).

	   The deserialize operator must look like:

	   bool operator()( const NodeT & );

	   It may be virtual or a function template.
	*/
	template <typename NodeT, typename DeserializableType>
	bool operator()( const NodeT & src, DeserializableType & dest ) const
	{
	    return dest.operator()( src );
	}
    };


    /**
       A default implementation for
       s11n_traits::cleanup_functor. Compatible implementations
       and specializations must follow the conventions defined for
       this class.

       This implementation is only suitable for cleaning up types
       which manage any child pointers which are assigned to them
       during deserialization. For example, std:: containers do
       not own their pointers, and therefor require a
       specialization of this type to clean them up.

       SerializableType must be a Serializable and may be
       pointer-qualified.

       The library manual goes into more detail about what this
       type is for.

       The operations of this class are declared as throw() because
       of their logical role in the destruction process, and destructors
       are not generally allowed to throw.
    */
    template <typename SerializableType>
    struct default_cleanup_functor
    {
	typedef typename type_traits<SerializableType>::type serializable_type;

	/**
	   Default implementation does nothing, though we
	   should arguably assign to a default-constructed
	   object.

	   Specializations must do any cleanup they need to
	   here. For example, container specializations must
	   a) deallocate any heap-based entries and b) empty
	   the list. Specializations are free to do a
	   default-assign, as mentioned above, but are not
	   required to. Specializations for containers must
	   recursively use s11n_trait::cleanup_functor for
	   the contained types, to ensure that they can properly
	   clean up containers holding other containers.
	*/
	void operator()( serializable_type & ) const throw()
	{
	}

    };

    /**
       EXPERIMENTAL!

       The tr namespace holds types for implementing the s11n_traits
       mechanism more modularly. Using these, we can customize parts
       of s11n_traits<T> without having provide all of the typedefs
       required by s11n_traits.

       Added in 1.2.8 and 1.3.2.
    */
    namespace tr {

	template <typename SerializableT, typename InterfaceT = SerializableT>
	struct s_sfunc
	{
	    typedef s11n::default_serialize_functor type;
	};

	template <typename SerializableT, typename InterfaceT = SerializableT>
	struct s_dfunc : s_sfunc<SerializableT,InterfaceT> {};

	template <typename SerializableT, typename InterfaceT = SerializableT>
	struct s_cleaner
	{
	    typedef s11n::default_cleanup_functor<InterfaceT> type;
	};

	template <typename SerializableT, typename InterfaceT = SerializableT>
	struct s_name
	{
	    static char const * name( const InterfaceT * /* instance_hint */ )
	    {
		return "unknown";
	    }
	};

	template <typename SerializableT, typename InterfaceT = SerializableT>
	struct s_factory
	{
	    typedef s_factory type;
	    InterfaceT * operator()( std::string const & cn ) const
	    {
		return s11n::cl::classload<InterfaceT>( cn.c_str() );
	    }
	};


    } // namespace

    /**
       s11n_traits encapsulates information about what
       type(s) are responsible for handling de/serialize
       operations for a given type, plus the factory for
       that type. It should be specialized to define
       various aspects of serialization for a given type.

       The interface shown here is the bare minimum which
       s11n_traits specializations must implement.
       Specializations may optionally add to the
       interface, but client code is discouraged from
       relying on any extensions.

       This type is stateless, and specializations are
       expected to be stateless (at least, they will
       be treated as if they are).

       Client code is not expected to need to use this
       type directly, except for purposes of plugging in
       their types into the s11n framework. More
       specifically, it is not expected that Serializables
       will use this type.

       Parameterized on:

       SerializableT: the base-most Serializable
       type. This is the base-most point of reference for
       classloading and "template typing". Subclasses of
       SerializableT are assumed to be handleable via the
       same de/serialize interface as SerializableT.

       InterfaceType is the base Serializable interface which
       SerializableT is assumed to subclass. The default is
       SerializableT. This type is required for cases where
       SerializableT wants to register with multiple Serializable
       interfaces.

       Changes from 1.0.x to 1.1.x:

       - Added class_name() member to replace the ::classname()
       family of code in 1.1.0.

       - cleanup_functor added in 1.1.3.

       - InterfaceType added in 1.2.7/1.3.1.
    */

    template <typename SerializableT, typename InterfaceType = SerializableT>
    struct S11N_EXPORT_API s11n_traits
    {
	/**
	   The s11n framework instantiates an s11n_traits
	   object at some points to allow the traits object
	   to do things like factory registration.
	*/
	s11n_traits(){}
	~s11n_traits(){}

	/**
	   The InterfaceType. Added in 1.3.1. This is needed
	   for cases where a Serializable wants to be registered
	   with several Serializable interfaces. If we don't
	   have this key then ODR violations happen on the second
	   and subsequent registration.
	*/
	typedef InterfaceType serializable_interface_type;

	/**
	   The type of object we want to [de]serialize.
	*/
	typedef SerializableT serializable_type;

	/**
	   Type which will be used to instantiate new objects
	   of serializable_type. It must implement:

	   serializable_type * operator()( const std::string & classname ) const;

	   It is expected to return, polymorphically if
	   possible, a new serializable_type on success or 0
	   on failure.

	   The default factory_type works with types
	   registered via the s11n::cl::classload()
	   family of functions.
	*/
	typedef typename tr::s_factory<serializable_type,serializable_interface_type>::type factory_type;

	/**
	   Functor type implementing serialize code.

	   Must implement:

	   bool operator()( SomeNodeType & dest, const base_type & src ) const;

	*/
	typedef typename tr::s_sfunc<serializable_type,serializable_interface_type>::type serialize_functor;

	/**
	   Functor type implementing deserialize code.

	   Must implement:

	   bool operator()( const SomeNodeType & src, base_type & dest ) const;
	*/
	typedef typename tr::s_dfunc<serializable_type,serializable_interface_type>::type deserialize_functor;

	/**
	   This type is used to clean up a partial deserialized object
	   on error. If this functor, and all specializations it calls,
	   do their part, we can make much better exception guarantees,
	   theoretically avoiding any leaks due to exceptions thrown
	   during deserialization.

	   cleanup_functor must follow the conventions laid out by
	   s11n::default_cleanup_functor<serializable_type>.
	*/
	typedef typename tr::s_cleaner<serializable_type,serializable_interface_type>::type cleanup_functor;

	/**
	   As of s11n 1.1, specializations must define the
	   class_name() function. This implementation returns
	   a useless, unspecified class name. Specializations
	   must return the class name of serializable_type,
	   preferably polymorphically (polymorphic naming is
	   unfortunately not possible without some client-side
	   help).

	   instance_hint is a HINT to this class as to the
	   actual instance we want the name for, and may be
	   0. It is provided so that class hierarchies which
	   have virtual functions like className() can make
	   those available to the core library via s11n_traits
	   specializations.

	   Specializations MUST accept 0 as a valid
	   instance_hint value are are NEVER REQUIRED to pay
	   any attention to instance_hint. The default
	   implementation does nothing with it.

	   Design notes:

	   - It really should take a default value of 0 for
	   instance_hint, but the idea of relying on a default
	   value, considering things like how template
	   specializations should define them and subclassing
	   (though that is not an issue *here*), gives me the
	   willies. Too much room for error there.

	   - Also, we could probably argue that it should
	   return a const string.

	   - We could argue that it should return an empty string
	   instead of a useless one. i don't want to to generate
	   output which might break parsers, though.
		   
	*/
	static const std::string class_name( const serializable_interface_type * hint )
	{
	    return tr::s_name<serializable_type,serializable_interface_type>::name(hint);
	}

    }; // end s11n_traits<>

//     template <typename SerializableT>
//     struct S11N_EXPORT_API s11n_traits<SerializableT> : s11n_traits<SerializableT,SerializableT>
//     {};


    /**
       A general specialization to treat (T*) as (T) for s11n_traits
       purposes. This is necessary for some template argument resolution
       to work as desired.

       Added in 1.1.3.
    */
    template <typename T>
    struct S11N_EXPORT_API s11n_traits<T *> : public s11n_traits<T> {};


}  // namespace s11n


#endif // s11n_S11N_TRAITS_HPP_INCLUDED
