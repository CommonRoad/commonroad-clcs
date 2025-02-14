#ifndef s11n_net_s11nlite_CLIENT_API_HPP_INCLUDED
#define s11n_net_s11nlite_CLIENT_API_HPP_INCLUDED 1
////////////////////////////////////////////////////////////////////////
//    License: Do As You Damned Well Please
//    Author: stephan@s11n.net

#include <memory> // auto_ptr
#include <iterator> // insert_interator<>

#include <s11n.net/s11n/s11n.hpp> // the whole s11n framework
#include <s11n.net/s11n/io/data_node_io.hpp> // s11n::io::data_node_serializer interface
#include <s11n.net/s11n/io/serializers.hpp> // create_serializer()

namespace s11nlite {

	/**
	   client_api is an abstraction of the 1.0.x s11nlite
	   interface (GAWD, NOT ANOTHER ABSTRACTION LAYER!), providing
	   an interface which works with all compatible S11n Node
	   types.

	   The intention of this class is for projects to have easy
	   access to an s11nlite-like interface for their own node
	   types. s11nlite, as of s11n 1.2, will be based off of
	   this type.

	   Another use of this class is in conjunction with
	   s11n::fac::instance_hook: clients may specialize that type
	   to make this type's instance() function return a custom
	   client_api object. The intention of this feature is to allow
	   clients to extend the s11nlite interface "from inside" while
	   allowing client code to keep using the s11nlite interface.
	   This allows, amongst other things, extending s11nlite
	   to support special i/o channels (as provided, e.g.,
	   by pclasses.com's libraries) without actually touching
	   s11nlite. See http://s11n.net/ps11n/ for where this is
	   headed...

	   Particularly observant users might notice that many of this
	   type's functions which "probably should be" const are not
	   const. This is because subclasses are encouraged to add
	   features behind the basic API, and many would not be
	   practical without a non-const object. As an example, a
	   subclass might want to do DLL lookups and may need to
	   modify internal tables when it does so. Template functions
	   which can be made const are const. This would appear to be
	   inconsistent, but it is in fact me trying to make up for
	   more of the API not being const.

	   The NodeType parameterized type must be compatible with the
	   s11n's "Data Node" type conventions (e.g.,
	   s11n::s11n_node).


	   Clients may notice that not much of this API is
	   virtual. That's not because i'm anal, but because most of
	   the API is template functions, and those can't be virual.
	   There is a theoretical solution, for those wanting more
	   control: subclass this type with another template type:
	   serializable_api<SerializableType>. Then you can add virtual
	   functions which take the place of the templated members.
	   Not sure if overloading rules will let you get away with
	   it, but it sounds halfway reasonable.


	   TODOs:

	   - Figure out how best to accomodate the
	   serializer_interface type as a template parameter. Right
	   now that's not feasible because the API depends too much on
	   the s11n::io namespace in general.
	*/
	template <typename NodeType>
	class client_api
	{
	private:
		/** Class name of preferred serializer type. */
		std::string m_serclass;
	public:

		client_api() : m_serclass(s11n_S11NLITE_DEFAULT_SERIALIZER_TYPE_NAME /* defined in s11n_config.hpp */ )
		{
		}

		explicit client_api( const std::string & default_serializer_class ) : m_serclass(default_serializer_class)
		{
		}

		virtual ~client_api()
		{
		}

		/**
		   node_type is the type used to store/load a Serializable
		   object's data.
		*/
		typedef NodeType node_type;

		/** The s11n::node_traits type for node_type. */
		typedef s11n::node_traits<node_type> node_traits;


		/**
		   This is the base-most type of the serializers used by s11nlite clients.
		*/
		typedef s11n::io::data_node_serializer<node_type> serializer_interface;


		/**
		   Returns create_serialize( serializer_class() ).
		   The caller owns the returned pointer.
		*/
		inline serializer_interface * create_serializer()
		{
			return this->create_serializer( this->serializer_class() );
		}

		/**
		   Returns a new instance of the given serializer class, or 0
		   if one could not be loaded. classname must represent a subtype
		   of serializer_interface.

		   The caller owns the returned pointer.

		   You can also pass a serializer's cookie here, and that
		   should return the same thing as it's class name would.

		   The internally-supported serializers all register a
		   "friendly form" of the name, an alias registered
		   with their classloader. Passing either this name or
		   the cookie of the Serializer should return the same
		   thing as the classname itself would.

		   Subclasses may do, e.g., lookups for
		   externally-linked serializers.
		*/
		virtual serializer_interface *
		create_serializer( const std::string & classname )
		{
			return ::s11n::io::create_serializer<node_type>( classname );
		}


		/**
		   Sets the current Serializer class used by s11nlite's
		   create_serializer(). Pass it a class name, or one of
		   the convenience names, e.g.:

		   compact, funtxt, funxml, simplexml, parens, wesnoth,
		   and expat (if your s11n was built with it).

		   You may use provides_serializer() to check for the
		   existence of a class. This function does no
		   validation of classname: this is delayed until
		   save/load-time, to allow for DLL lookups during the
		   normal classloading process (as opposed to
		   duplicating that check here).
		*/
		inline void serializer_class( const std::string & classname )
		{
			this->m_serclass = classname;
		}


		/**
		   Gets the name of the current Serializer type.
		*/
		inline std::string serializer_class() const
		{
			return this->m_serclass;
		}

		/**
		   Returns true if key can be used to create a Serializer
		   object via a call to serializer_class(). Subclasses
		   may do additional work here, like look up DLLs, which is
		   why the function is not const.
		*/
		virtual bool provides_serializer( const std::string & key )
		{
			typedef ::s11n::fac::factory_mgr<serializer_interface> FacMgr;
			return FacMgr::instance().provides( key );
		}


		/**
		   See s11n::serialize().
		*/
		template <typename SerializableType>
		inline bool serialize( node_type & dest, const SerializableType & src ) const
		{
			return s11n::serialize<node_type,SerializableType>( dest, src );
		}

		/**
		   See s11n::serialize_subnode().
		*/
		template <typename SerializableType>
		inline bool serialize_subnode( node_type & dest, const std::string & subnodename, const SerializableType & src ) const
		{
			// todo: consider custom-implementing to go through the local de/serialize() funcs.
			// The end result would be the same, since those funcs aren't virtual and we know
			// in advance that they will behave exactly the same as this approach.
			// return s11n::serialize_subnode<node_type,SerializableType>( dest, subnodename, src );
			// Nevermind... we can't. If we do, then (SerializableType*) is not forwarded properly
			// without some additional type juggling.
			std::unique_ptr<node_type> n( node_traits::create( subnodename ) );
			if( this->serialize<SerializableType>( *n, src ) )
			{
                               node_traits::children(dest).push_back( n.release() );
                               return true;
			}
			return false;

		}

       
		/**
		   Saves the given node to the given ostream using the default
		   serializer type.

		   Returns true on success, false on error.

		   ONLY use this for saving root nodes!
		*/
		virtual bool save( const node_type & src, std::ostream & dest )
		{
			std::unique_ptr<serializer_interface> s(this->create_serializer());
			if( s.get() )
			{
			    return s->serialize( src, dest );
			}
			else
			{
			    return false;
			}
		}

		/**
		   Saves the given node to the given filename using the default
		   serializer type.

		   Returns true on success, false on error.

		   ONLY use this for saving root nodes!

		   Subclasses are free to interpret the filename
		   however they like, e.g., as a URL or database
		   record name.
		*/
		virtual bool save( const node_type & src, const std::string & filename )
		{
			std::unique_ptr<serializer_interface> s(this->create_serializer());
			return s.get()
				? s->serialize( src, filename )
				: false;
			// Maintenance note: we used to just convert
			// filename to a stream and call this->save(
			// src, ostr ), but this poses a problem with
			// serializers, like mysql_serializer, which
			// behave differently in the face of streams
			// and filenames. Currently (1.1.3)
			// mysql_serializer is the only one which
			// needs this distinction (because it can't
			// write db records to a stream). The experimental
			// ps11n support also distinguishe, as it supports
			// serialization over ftp URLs.
		}


		/**
		   Saves the given Serializable to the given ostream using the default
		   serializer type.
		   Returns true on success, false on error.

		   ONLY use this for saving root nodes!

		   Subclassers: this function serializes src to a node and then
		   calls save(node,dest).
		*/
		template <typename SerializableType>
		inline bool save( const SerializableType & src, std::ostream & dest )
		{
			node_type n;
			if( this->serialize( n, src ) )
			{
			    return this->save( n, dest );
			}
			else
			{
			    return false;
			}
		}
		/**

		   Saves the given Serializable to the given filename using the default
		   serializer type.
        
		   Returns true on success, false on error.

		   ONLY use this for saving root nodes!

		   Subclassers: this function serializes src to a node and then
		   calls save(node,dest).
		*/
		template <typename SerializableType>
		bool save( const SerializableType & src, const std::string & dest )
		{
// See save(node,string) for why we don't wan this:
// 			typedef std::unique_ptr<std::ostream> AP;
// 			AP os = AP( s11n::io::get_ostream( dest ) );
// 			if( ! os.get() ) return 0;
// 			return this->save( src, *os );
			node_type n;
			return this->serialize( n, src )
				? this->save( n, dest )
				: false;
		}

		/**
		   Tries to load a node from the given filename.

		   The caller owns the returned pointer.

		   Subclasses are free to interpret the filename
		   however they like, e.g., as a URL or database
		   record name.
		*/        
		virtual node_type * load_node( const std::string & src )
		{
			return s11n::io::load_node<node_type>( src );
		}


		/**
		   Tries to load a node from the given input stream.

		   The caller owns the returned pointer, which may be
		   0.

		   Only usable for loading ROOT nodes.
		*/
		virtual node_type * load_node( std::istream & src )
		{
			return s11n::io::load_node<node_type>( src );
		}



		/**
		   Returns s11n::deserialize<node_type,SerializableType>(src).

		   Caller owns the returned pointer, which may be 0.

		   Note that this function is non-const because deserialization
		   may indirectly classload other types or affect this object.

		   See the one-argument variant of s11n::deserialize() for
		   IMPORTANT error-handling considerations for this function.
		*/
		template <typename SerializableType>
		inline SerializableType * deserialize( const node_type & src ) const
		{
			return s11n::deserialize<node_type,SerializableType>( src );
		}



		/**
		   See s11n::deserialize().
		*/
		template <typename DeserializableT>
		inline bool deserialize( const node_type & src, DeserializableT & target ) const
		{
			return s11n::deserialize<node_type,DeserializableT>( src, target );
		}


		/**
		   See s11n::deserialize_subnode().
		*/
		template <typename DeserializableT>
		inline bool deserialize_subnode( const node_type & src, const std::string & subnodename, DeserializableT & target ) const
		{
			return s11n::deserialize_subnode< node_type, DeserializableT >( src, subnodename, target );
		}

		/**
		   See s11n::deserialize_subnode().
		*/
		template <typename DeserializableT>
		inline DeserializableT * deserialize_subnode( const node_type & src, const std::string & subnodename ) const
		{
			return s11n::deserialize_subnode< node_type, DeserializableT >( src, subnodename );
		}


		/**
		   Tries to load a data_node from src, then deserialize that
		   to a SerializableType.
		*/
		template <typename SerializableType>
		inline SerializableType * load_serializable( std::istream & src )
		{
			std::unique_ptr<node_type> n( this->load_node( src ) );
			return n.get()
				? this->deserialize<SerializableType>( *n )
				: 0;
		}

		/**
		   Overloaded form which takes a file name.

		   1.1.3: removed the never-used 2nd parameter.

		   Subclassers: the loading is done via load_node(),
		   so you can intercept the src string there.
		*/
		template <typename SerializableType>
		inline SerializableType * load_serializable( const std::string & src )
		{
			std::unique_ptr<node_type> node( this->load_node(src) );
			return node.get()
				? this->deserialize<SerializableType>( *node )
				: 0;
		}

		/**
		   See s11n::s11n_clone().
		*/
		template <typename SerializableType>
		inline SerializableType * clone( const SerializableType & tocp ) const
		{
			return s11n::s11n_clone<node_type,SerializableType>( tocp );
		}

		/**
		   See s11n::s11n_cast().
		*/
		template <typename Type1, typename Type2>
		inline bool cast( const Type1 & t1, Type2 & t2 ) const
		{
			// Hmmmm.... should we jump directly to s11n_cast() or perform that same
			// operation using our de/serialize() API????
			// They aren't necessarily equivalent, and the inconsistency might come
			// to light one day. i can't imagine a use case, but there's gotta
			// be one.
			// node_type n;
			// return this->serialize<Type1>( n, t1 ) && this->deserialize<Type2>( n, t2 );
			return ::s11n::s11n_cast<node_type,Type1,Type2>( t1, t2 );
		}

		/**
		   Returns a shared instance of this class. The object
		   itself is post-main() safe (on most systems),
		   meaning if this function is called after the shared
		   object is destroy (after main() exits), it "should"
		   actually behave properly be re-instantiating the
		   object in-place. See s11n/phoenix.hpp for details
		   and caveats. In the worst case, it's no less safe than
		   any other post-main() object.
		*/
		static client_api<node_type> &
		instance()
		{
			return ::s11n::Detail::phoenix< client_api<node_type> >::instance();
		}

	}; // client_api<> class 

} // namespace s11nlite

#endif // s11n_net_s11nlite_CLIENT_API_HPP_INCLUDED
