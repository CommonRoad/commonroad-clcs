#ifndef S11N_LITE_H_INCLUDED
#define S11N_LITE_H_INCLUDED 1

////////////////////////////////////////////////////////////////////////
// headers which want to check for s11nlite's inclusion should check
// for:
#define s11n_S11NLITE_INCLUDED 1
// That's "the official way" to know if s11nlite is avaliable.
////////////////////////////////////////////////////////////////////////

/**
   s11nlite is a simplified subset of the s11n interface, combining
   the APIs from the core and the s11n::io namespace into a single
   API. It refines the s11n interface to almost-minimal, while still
   leaving the full power of the underlying framework at clients'
   disposals if they need it. It also provides a binding to the
   default i/o implementation (the s11n::io-related Serializers),
   meaning that clients don't have to deal with file formats at all,
   and almost never even need to deal with the Serializer interface.

   It is completely compatible with the core s11n framework, but
   directly provides only a subset of the interface: those operations
   common to most client-side use-cases, and those functions where
   s11nlite can eliminate one of the required template parameters
   (s11n's conventional NodeType argument, which s11nlite hard-codes
   in its note_type typedef).

   This implementation can easily be used as a basis for writing
   custom client-side, s11n-based serialization frameworks.

   Suggestions for including specific features into this interface are
   of course welcomed.

   Common conventions concerning this API:

   - node_type must conform to s11n's S11n Node conventions.

   - node_traits is equivalent to s11n::node_traits<node_type>.

   - Serializers usable by this framework must be classloadable via
     the serializer_interface classloader (including all that this
     implies).


     As of version 1.1, s11nlite is primarily a wrapper around the
     type s11nlite::client_api<>. Many of s11nlite's functions
     internally use a client_api instance to do their work. This
     allows clients to swap out most of the s11nlite implementation
     with their own at runtime, while still allowing all s11nlite
     clients to use the same front-end API.


     License: Do As You Damned Well Please

     Author: stephan@s11n.net
*/
namespace s11nlite { /** here to please doxygen :( */ }

#include <memory> // auto_ptr
#include <iterator> // insert_interator<>

#include <s11n.net/s11n/s11n.hpp> // the core s11n framework
#include <s11n.net/s11n/io/data_node_io.hpp> // s11n::io::data_node_serializer class
#include <s11n.net/s11n/client_api.hpp> // the "real" implementation for most of s11nlite.

//Added by Damien to make Windows compile work
#include <s11n.net/s11n/export.hpp> // S11N_EXPORT_API


namespace s11nlite {



	/**
	   client_interface defines the interface used/returned by the
	   instance() functions. By subclassing this type and
	   reimplmenting the appropriate virtual functions, a client-supplied
	   instance of this type can be plugged in as the primary s11nlite API
	   handler. This allows, for example, transparently adding compression
	   or network support to s11nlite.
	*/
	typedef client_api<s11n::s11n_node> client_interface;

        /**
           node_type is the type used to store/load a Serializable
           object's data.

	   FYI: node_type might change to s11n::s11n_node in 1.1,
	   because that implementation is a tiny bit more lightweight
	   than s11n::data_node, and was designed specifically with
	   node_traits in mind (data_node came much earlier than
	   either of them).  If you only use node_traits to access
	   nodes' data then your code will be oblivious to this change
	   (but will need a recompile).
        */
        typedef client_interface::node_type node_type;

        /** The s11n::node_traits type for node_type. */
        typedef client_interface::node_traits node_traits;

        /**
           This is the base-most type of the serializers used by s11nlite.
        */
        typedef client_interface::serializer_interface serializer_interface;

	/**
	   Returns the client_interface object used by the s11nlite
	   API. There is guaranteed to return *some* object, except
	   possibly post-main() (where all bets are off).

	   See instance(client_interface*) for more details.
	*/
   //Added by Damien to make Windows compile work
	S11N_EXPORT_API client_interface & instance();

	/**
	   Sets the client_interface object used by the s11nlite
	   API. Ownership of newinstance IS NOT transfered. Passing
	   NULL will cause it to revert to the default instance.
	   newinstance must live as long as any client is using
	   s11nlite, and when newinstance is destroyed, 0 must be
	   passed to this function to disassociate the object from the
	   library.
	*/
	void instance( client_interface * newinstance );

        /**
           Returns a new instance of the default serializer class.

           The caller owns the returned pointer.

         */
        serializer_interface * create_serializer();

        /**
           Returns a new instance of the given serializer class, or 0
           if one could not be loaded. classname must represent a subtype
           of serializer_interface.

           The caller owns the returned pointer.

           You can also pass a serializer's cookie here, and that
           should return the same thing as its class name would.

           The internally-supported serializes all support a "friendly
           form" of the name, an alias registered with their
           classloader. Passing either this name or the cookie of the
           Serializer should return the same thing as the classname
           itself would.

           Short-form names of internally-supported Serializers:

           - compact (also called 51191011)
           - expat (IF s11n was built with libexpat support)
           - funtxt
           - funxml
           - parens
           - simplexml
           - wesnoth

           Note that the short-form name is always the same as that of
           the underlying Serializer class, minus the _serializer
           suffix. This is purely a convention, and not a rule. This command
	   will use s11n::plugin support, if available, to dynamically
	   load new serializers. Simply name the DLL the same as the class,
	   without any namespace part in the filename, and your platform-specific
	   DLL file extension (e.g. .dll, .so, or .dynlib).
         */
        serializer_interface *
        create_serializer( const std::string & classname );


        /**
           Sets the current Serializer class used by s11nlite's
           create_serializer(). Pass it a class name, or one of the
           convenience names listed in create_serializer(string).
        */
        void serializer_class( const std::string & );

        /**
           Gets the name of the current Serializer type.
        */
        std::string serializer_class();


        /**
           A non-const overload. Ownership is not modified by calling
           this function: normally the parent node owns it.
        */
        node_type *
        find_child( node_type & parent,
                    const std::string subnodename );

        /**
           Equivalent to s11n::find_child_by_name( parent, subnodename ).
        */
        const node_type *
        find_child( const node_type & parent,
                    const std::string subnodename );

        /**
           See s11n::serialize().
        */
        template <typename SerializableType>
        bool serialize( node_type & dest,
                        const SerializableType & src )
        {
                return instance().template serialize<SerializableType>( dest, src );
        }

        /**
           See s11n::serialize().
        */
        template <typename SerializableType>
        bool serialize_subnode( node_type & dest,
                                const std::string & subnodename,
                                const SerializableType & src )
        {
		return instance().template serialize_subnode<SerializableType>( dest, subnodename, src );
        }

       
        /**
        Saves the given node to the given ostream using the default
        serializer type.

        Returns true on success, false on error.

        ONLY use this for saving root nodes!
        */
        bool save( const node_type & src, std::ostream & dest );

        /**
        Saves the given node to the given filename using the default
        serializer type.

        Returns true on success, false on error.

        ONLY use this for saving root nodes!
        */
        bool save( const node_type & src, const std::string & filename );

        /**
        Saves the given Serializable to the given ostream using the default
        serializer type.

        Returns true on success, false on error.

        ONLY use this for saving root nodes!
        */
        template <typename SerializableType>
        bool save( const SerializableType & src, std::ostream & dest )
        {
		return instance().template save<SerializableType>( src, dest );
        }
        /**
        Saves the given Serializable to the given filename using the default
        serializer type.
        
        Returns true on success, false on error.

        ONLY use this for saving root nodes!
        */
        template <typename SerializableType>
        bool save( const SerializableType & src, const std::string & dest )
        {
		return instance().template save<SerializableType>(src,dest);
        }

        /**
           Tries to load a node from the given filename.

           The caller owns the returned pointer.
         */        
        node_type * load_node( const std::string & src );

        /**
           Tries to load a node from the given input stream.

           The caller owns the returned pointer.

           Only usable for loading ROOT nodes.
         */        
        node_type * load_node( std::istream & src );


        /**
	   Tries to load a node_type from the given source.  On
	   success, dest is set to the contents of that source and
	   true is returned, otherwise false is returned and dest is
	   untouched.

	   Added in versions 1.3.1 and 1.2.8.
	*/
        bool load_node( const std::string & src, node_type & dest );

        /**
	   Overloaded form taking an input stream instead of a string.

	   Added in versions 1.3.1 and 1.2.8.
	*/
        bool load_node( std::istream & src, node_type & dest );

        /**
	   See s11n::deserialize().

           ACHTUNG: if you are using both s11n and s11nlite
           namespaces this function will be ambiguous with one provided
           in the namespace s11n. You must then qualify it
           with the namespace of the one you wish to use.
        */
        template <typename SerializableType>
        SerializableType * deserialize( const node_type & src )
        {
                return instance().template deserialize<SerializableType>( src );
        }



        /**
           Tries to deserialize src into target. Returns true on
           success. If false is returned then target is not guaranteed
           to be in a useful state: this depends entirely on the
           object (but, it could be argued, it it was in a useful
           state its deserialize operator would have returned true!).

	   DO NOT PASS A POINTER TYPE TO THIS FUNCTION. It will not do
	   what you want it to, and it will likely cause a leak. If
	   you want to directly deserialize to a pointer, use the
	   s11n::deserialize<>() overload which takes a reference to a
	   pointer.
        */
        template <typename DeserializableT>
        bool deserialize( const node_type & src, DeserializableT & target )
        {
                return instance().template deserialize<DeserializableT>( src, target );
        }


        /**
           Exactly like deserialize(), but operates on a subnode of
           src named subnodename. Returns false if no such file is
           found.
         */
        template <typename DeserializableT>
        bool deserialize_subnode( const node_type & src,
                                  const std::string & subnodename,
                                  DeserializableT & target )
        {
                return instance().template deserialize_subnode<DeserializableT>( src, subnodename, target );
        }

        /**
           Exactly like deserialize(), but operates on a subnode of
           src named subnodename. Returns 0 if no such file is
           found.
         */
        template <typename DeserializableT>
        DeserializableT * deserialize_subnode( const node_type & src,
                                               const std::string & subnodename )
        {
                return instance().template deserialize_subnode<DeserializableT>( src, subnodename );
        }


        /**
           Tries to load a data_node from src, then deserialize that
           to a SerializableType.
        */
        template <typename SerializableType>
        SerializableType * load_serializable( std::istream & src )
        {
		return instance().template load_serializable<SerializableType>( src );
        }

        /**
           Overloaded form which takes a file name.

	   1.1.3: removed the never-used 2nd parameter.
        */
        template <typename SerializableType>
        SerializableType * load_serializable( const std::string & src )
        {
		return instance().template load_serializable<SerializableType>( src );
        }

        /**
           See s11n::s11n_clone().

	   This function was renamed from clone() in version 1.1.
        */
        template <typename SerializableType>
        SerializableType * s11n_clone( const SerializableType & tocp )
        {
		return instance().template clone<SerializableType>( tocp );
        }

        /**
           See s11n::s11n_cast().
        */
        template <typename Type1, typename Type2>
        bool s11n_cast( const Type1 & t1, Type2 & t2 )
        {
		return instance().template cast<Type1,Type2>(t1,t2);
        }



	/**
	   A binary functor to save s-nodes and Serializables
	   using s11nlite::save().

	   Added in version 1.1.3.
	*/
	struct save_binary_f
	{
		typedef ::s11nlite::node_type node_type;

		/**
		   Returns save(src,dest).
		*/
		inline bool operator()( node_type const & src, std::string const & dest ) const
		{
			return ::s11nlite::save( src, dest );
		}

		/**
		   Returns save(src,dest).
		*/
		inline bool operator()( node_type const & src, std::ostream & dest ) const
		{
			return ::s11nlite::save( src, dest );
		}

		/**
		   Returns save(src,dest).
		*/
		template <typename SerT>
		inline bool operator()( SerT const & src, std::string const & dest ) const
		{
			return ::s11nlite::save( src, dest );
		}

		/**
		   Returns save(src,dest).
		*/
		template <typename SerT>
		inline bool operator()( SerT const & src, std::ostream  & dest ) const
		{
			return ::s11nlite::save( src, dest );
		}


	};

	/**
	   A functor which forwards to s11nlite::save(node_type,string).

	   See save_stream_unary_f for notes about the parent classes.
	   Those same notes apply here.

	   Added in version 1.1.3.
	*/
	struct save_string_unary_f : ::s11n::serialize_unary_serializable_f_tag,
				     ::s11n::serialize_unary_node_f_tag
	{
		typedef ::s11nlite::node_type node_type;
		const std::string destination;
		/**
		   Specifies that operator() should send output to the
		   given resource name (normally, but not always, a
		   filename).
		*/
		explicit save_string_unary_f( std::string const & s ) : destination(s)
		{}

		/**
		   Returns s11nlite:;save( src, this->destination ).
		*/
		bool operator()( node_type const & src ) const
		{
			return ::s11nlite::save( src, this->destination );
		}

		/**
		   Returns s11nlite:;save( src, this->destination ).
		*/
		template <typename SerT>
		bool operator()( SerT const & src ) const
		{
			return ::s11nlite::save( src, this->destination );
		}
	};


	/**
	   A unary functor which forwards to
	   s11nlite::save(node_type|SerializableT,ostream).

	   Note that while this type conforms to two s11n-standard
	   tags (its parent classes), it is not *really* intended to
	   be used as a normal part of a serialization algorithm. That
	   said, that approach may indeed turn out to have some
	   interesting uses. It certainly has some pitfalls, in any
	   case, so don't blythely do it.

	   Added in version 1.1.3.
	*/
	struct save_stream_unary_f : ::s11n::serialize_unary_serializable_f_tag,
				     ::s11n::serialize_unary_node_f_tag
	{
		typedef ::s11nlite::node_type node_type;
		std::ostream & stream;
		/**
		   Specifies that operator() should send output to the
		   given stream.
		*/
		explicit save_stream_unary_f( std::ostream & os ) : stream(os)
		{}

		/**
		   Returns s11nlite:;save( src, this->stream ).
		*/
		bool operator()( node_type const & src ) const
		{
			return ::s11nlite::save( src, this->stream );
		}

		/**
		   Returns s11nlite:;save( src, this->stream ).
		*/
		template <typename SerT>
		bool operator()( SerT const & src ) const
		{
			return ::s11nlite::save( src, this->stream );
		}
	};


	/**
	   An "implementation detail" nullary functor type to simplify
	   implementations of save_xxx_nullary_f. It is not intended
	   for standalone use, but as a base type for
	   save_xxx_nullary_f functors.

	   T must be one of:

	   - s11nlite::node_type

	   - A non-cvp qualified Serializable type.

	   OutputT is expected to be one of:

	   - [const] std::string. NOT a reference, because we're
	   lazily evaluating and don't want to step on a dead
	   temporary.

	   - (std::ostream &)

	   If s11nlite::save() is ever overloaded, e.g., to take your
	   own custom output destination type, then that type will
	   also theoretically work here

	   See save() for important notes about when you should NOT
	   use this. In particular, this functor should not normally
	   be used with std::for_each(), as save() expects to store
	   ONE object in each target. Loading objects saved this way
	   won't work: only the first one is likely to be read by
	   load-time. Exceptions to this caveat include network
	   streams or debug channels.

	   Added in version 1.1.3.

	*/
	template <typename T,typename OutputT>
	struct save_nullary_base_f
	{
		T const & source;
		OutputT destination;
		/**
		   Both src and dest are expected to outlive this
		   object, unless of source OutputT is a value type,
		   in which case we copy dest at ctor time, instead of
		   taking a (const &), to avoid us accidentally storing
		   a reference to a temporary which probably won't
		   exist when operator() is called.
		*/
		save_nullary_base_f( T const & src, OutputT dest )
			: source(src), destination(dest)
		{}
		/**
		   Returns ::s11nlite::save( this->source, this->destination ).
		*/
		inline bool operator()() const
		{
			return ::s11nlite::save( this->source, this->destination );
		}
	};

	/**
	   A nullary functor forwarding to s11nlite::save(node,string).
	*/
	struct save_node_string_nullary_f : save_nullary_base_f< ::s11nlite::node_type, const std::string >
	{
		typedef save_nullary_base_f< ::s11nlite::node_type, const std::string > parent_t;
		typedef ::s11nlite::node_type node_type;
		/**
		   See the notes in the base type API for requirements
		   of src and dest.
		*/
		save_node_string_nullary_f( node_type const & src, std::string const & dest )
			: parent_t( src, dest )
		{}

	};

	/**
	   A nullary functor forwarding to s11nlite::save(node,string).

	*/
	struct save_node_stream_nullary_f : save_nullary_base_f< ::s11nlite::node_type, std::ostream & >
	{
		typedef save_nullary_base_f< ::s11nlite::node_type, std::ostream & > parent_t;
		typedef ::s11nlite::node_type node_type;
		/**
		   See the notes in the base type API for requirements
		   of src and dest.
		*/
		save_node_stream_nullary_f( node_type const & src, std::ostream & dest )
			: parent_t( src, dest )
		{}
	};

	/**
	   A nullary functor forwarding to s11nlite::save(SerT,string).
	*/
	template <typename SerT>
	struct save_serializable_string_nullary_f : save_nullary_base_f< SerT, const std::string >
	{
		typedef save_nullary_base_f< SerT, const std::string > parent_t;
		/**
		   See the notes in the base type API for requirements
		   of src and dest.
		*/
		save_serializable_string_nullary_f( SerT const & src, std::string const & dest )
			: parent_t( src, dest )
		{}
	};

	/**
	   A nullary functor forwarding to s11nlite::save(Serializable,ostream).
	*/
	template <typename SerT>
	struct save_serializable_stream_nullary_f : save_nullary_base_f< SerT, std::ostream & >
	{
		/**
		   See the notes in the base type API for requirements
		   of src and dest.
		*/
		typedef save_nullary_base_f< SerT, std::ostream & > parent_t;
		save_serializable_stream_nullary_f( SerT const & src, std::ostream & dest )
			: parent_t( src, dest )
		{}
	};


	/**
	   Returns save_serializable_string_nullary_f<SerT>( src, dest ).
	*/
	template <typename SerT>
	inline save_serializable_string_nullary_f<SerT>
	save_nullary_f( SerT const & src, std::string const & dest )
	{
		return save_serializable_string_nullary_f<SerT>( src, dest );
	}

	/**
	   Returns save_serializable_stream_nullary_f<SerT>( src, dest ).
	*/
	template <typename SerT>
	inline save_serializable_stream_nullary_f<SerT>
	save_nullary_f( SerT const & src, std::ostream & dest )
	{
		return save_serializable_stream_nullary_f<SerT>( src, dest );
	}

	/**
	   Returns save_node_string_nullary_f( src, dest ).
	*/
	inline save_node_string_nullary_f
	save_nullary_f( node_type const & src, std::string const & dest )
	{
		return save_node_string_nullary_f( src, dest );
	}
	/**
	   Returns save_node_stream_nullary_f( src, dest ).
	*/
	inline save_node_stream_nullary_f
	save_nullary_f( node_type const & src, std::ostream & dest )
	{
		return save_node_stream_nullary_f( src, dest );
	}




	/**
	   A nullary functor to call s11nlite::load_node(istream&).

	   Added in version 1.1.3.
	*/
	struct load_node_stream_nullary_f
	{
		std::istream & stream;
		typedef ::s11nlite::node_type node_type;
		/**
		   Specifies that operator() should fetch input from the
		   given stream.
		*/
		explicit load_node_stream_nullary_f( std::istream & s ) : stream(s)
		{}

		/**
		   Returns s11nlite::load_node( this->stream ),

		   Caller owns the returned object, which may be 0.
		*/
		inline node_type * operator()() const
		{
			return ::s11nlite::load_node( this->stream );
		}
	};

	/**
	   A nullary functor to call s11nlite::load_node(string).
	*/
	struct load_node_nullary_string_f
	{
		typedef ::s11nlite::node_type node_type;
		const std::string resource;
		/**
		   Specifies that operator() should fetch input from
		   the given resource name (normally, but not always,
		   a filename).
		*/
		explicit load_node_nullary_string_f( std::string const & s ) : resource(s)
		{}

		/**
		   Returns s11nlite::load_node( this->resource ).

		   Caller owns the returned object, which may be 0.
		*/
		inline node_type * operator()() const
		{
			return ::s11nlite::load_node( this->resource );
		}
	};


	/**
	   Returns load_node_nullary_string_f(s).
	*/
	inline load_node_nullary_string_f
	load_node_nullary_f( std::string const & s )
	{
		return load_node_nullary_string_f(s);
	}

	/**
	   Returns load_node_stream_nullary_f(s).
	*/
	inline load_node_stream_nullary_f
	load_node_nullary_f( std::istream & s )
	{
		return load_node_stream_nullary_f(s);
	}
	

	/**
	   A unary functor to call s11nlite::load_node(string|stream).
	*/
	struct load_node_unary_f
	{

		/**
		   Returns s11nlite::load_node( src ).
		*/
		inline bool operator()( std::string const & src ) const
		{
			return ::s11nlite::load_node( src );
		}

		/**
		   Returns s11nlite::load_node( src ).
		*/
		inline bool operator()( std::istream & src ) const
		{
			return ::s11nlite::load_node( src );
		}
	};


} // namespace s11nlite

#endif // S11N_LITE_H_INCLUDED
