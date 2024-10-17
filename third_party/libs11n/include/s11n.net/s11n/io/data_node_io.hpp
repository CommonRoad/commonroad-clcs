#ifndef s11n_DATA_NODE_IO_H_INCLUDED
#define s11n_DATA_NODE_IO_H_INCLUDED

////////////////////////////////////////////////////////////////////////
// data_node_io.hpp
// some i/o interfaces & helpers for s11n
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////


#include <string>
#include <sstream>
#include <list>
#include <map>
#include <deque>
#include <iostream>
#include <memory>// auto_ptr

#include <cassert>
#include <typeinfo>



// #include <s11n.net/cl/cllite.hpp> // for opening DLLs

#include <s11n.net/s11n/phoenix.hpp> // phoenix class

#include <s11n.net/s11n/exception.hpp>
#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // COUT/CERR
#include <s11n.net/s11n/classload.hpp> // classloader()
#include <s11n.net/s11n/serialize.hpp> // unfortunately dep
#include <s11n.net/s11n/traits.hpp> // s11n_traits & node_traits

#include <s11n.net/s11n/export.hpp> // for exporting symbols to DLL

////////////////////////////////////////////////////////////////////////////////
// NO DEPS ON s11n_node.hpp ALLOWED!
////////////////////////////////////////////////////////////////////////////////


#define s11n_SERIALIZER_ENABLE_INTERACTIVE 0 /* an experiment. */

namespace s11n {

        namespace io {

		/**
		   Returns an output stream for the given file
		   name. Caller owns the returned pointer, which may
		   be 0.

		   The returned stream supports libzl and libbz2 if your
		   libs11n is built with libs11n_zfstream support, meaning
		   it can read files compressed with zlib/gzip or bz2lib.
		*/
		std::ostream * get_ostream( const std::string name );

		/**
		   Returns an input stream for the given file
		   name. Caller owns the returned pointer, which may
		   be 0.

		   The returned stream supports libzl and libbz2 if
		   your libs11n is built with libs11n_zfstream
		   support. That means that if
		   zfstream::compression_policy() is set, then the
		   returned string might be a compressing stream.

		   If ExternalData is false then name is assumed to be
		   a string containing input, and a string-reading stream
		   is returned.
		*/
		std::istream * get_istream( const std::string name, bool ExternalData = true );

                /**
                   Convenience function for grabbing the first line of a file.

                   If ExternalData == true then returns the first line of the
                   file, else returns up to the first newline of src.

		   See get_magic_cookie( istream & ) for notes on a minor functional
		   change introduced in version 1.2.1.
                */
                std::string get_magic_cookie( const std::string & src, bool ExternalData = true );

                /**
                   Convenience function for grabbing the first line of a
                   stream.

                   Returns the first line of the given stream, or an
                   empty string on error.

		   As of version 1.2.1, this function behaves slightly
		   differently than prior versions: the returned
		   string will be the first consecutive non-control
		   characters in the line.  This allows us to properly
		   read some binary formats which use a string
		   identifier as a magic cookie (e.g.  sqlite
		   databases). In this context "control characters"
		   are anything outside the range of ASCII values
		   [32..126]. This change "shouldn't" affect any
		   pre-1.2.1 behaviours, which were never tested/used
		   with binary file formats.
                */
                std::string get_magic_cookie( std::istream & is );

#if s11n_SERIALIZER_ENABLE_INTERACTIVE
		struct progress_reporter
		{
			progress_reporter() {}
			virtual ~progress_reporter() {}
			virtual void operator()( size_t pos, size_t total ) = 0;
		};
#endif // s11n_SERIALIZER_ENABLE_INTERACTIVE


                /**
                   data_node_serializer provides an interface for
                   saving/loading a given abstract data node type
                   to/from streams.

                   It is designed for containers which comply with
                   s11n's Data Node interface and conventions.


                   Conventions:

                   Must provide:

                   typedef NodeT node_type

                   Two de/serialize functions, following the
                   stream-based interface shown here (filename-based
                   variants are optional, but convenient for clients).


		   Potential TODOs for 1.3/1.4:

		   - Add cancel() and cancelled() to set/query the
		   read state. This is to support Cancel operations in
		   UIs which load slow-loading (sqlite3) formats or
		   large sets and want to safely cancel. Once
		   cancelled, a read is not restartable (or this is
		   not guaranteed). All we can really do is provide a
		   flag and conventions and hope implementations
		   respect them.

		   - Provide some sort of progress feedback mechanism,
		   at least for reading, again to support users of
		   slow Serializers. This is complicated by the
		   unknown-read-size nature of Serializers.
                */
                template <typename NodeT>
                class S11N_EXPORT_API data_node_serializer
                {
                public:
                        /**
                           The underlying data type used to store
                           serialized data.
                        */
                        typedef NodeT node_type;


                        data_node_serializer()
                        {
                                this->magic_cookie( "WARNING: magic_cookie() not set!" );
                                // ^^^ subclasses must do this.
				typedef ::s11n::node_traits<node_type> NTR;
				NTR::name( this->metadata(), "serializer_metadata" );
                                // this->metadata().name( "serializer_metadata" );
				using namespace s11n::debug;
				S11N_TRACE(TRACE_CTOR) << "data_node_serialier()\n";

                        };
                        virtual ~data_node_serializer()
			{
				using namespace s11n::debug;
				S11N_TRACE(TRACE_DTOR) << "~data_node_serialier() ["<<this->magic_cookie()<<"]\n";
			}


                        /**
                           A convenience typedef, mainly for subclasses.
                        */
                        typedef std::map<std::string,std::string> translation_map;

                        /**
                           Returns a map intended for use with
                           ::s11n::io::strtool::translate_entities().
                           
                           The default implementation returns an empty map.
                           
                           Subclasses should override this to return a translation
                           map, if they need one. The default map is empty.

                           Be aware that this may very well be called
                           post-main(), so subclasses should take that into
                           account and provide post-main()-safe maps!
                        */
                        virtual const translation_map & entity_translations() const
                        {
                                typedef ::s11n::Detail::phoenix<translation_map,data_node_serializer<node_type> > TMap;
                                return TMap::instance();
                        }



                        /**
                           Must be implemented to format node_type to the given ostream.

                           It should return true on success, false on error.

                           The default implementation always returns false.

                           Note that this function does not use
                           s11n::serialize() in any way, and is only
                           coincidentally related to it.
                        */
                        virtual bool serialize( const node_type & /*src*/, std::ostream & /*dest*/ )
                        {
                                return false;
                        }

                       /**
                           Overloaded to save dest to the given filename.

                           The default implementation treats destfile
                           as a file name and passes the call on to
                           serialize(node_type,ostream).  The output
                           file is compressed if zfstream::compression_policy()
                           has been set to enable it.

                           Returns true on success, false on error.

                           This function is virtual so that
                           Serializers which do not deal with
                           i/ostreams (e.g., those which use a
                           database connection) can override it to
                           interpret destfile as, e.g., a
                           database-related string (e.g., connection,
                           db object name, or whatever).

			   Fixed in 1.0.2: returns false when destfile
			   is empty.

			   Fixed in 1.2.6: if this->serialize(src,outstream) throws then
			   this function no longer leaks.
                        */
                        virtual bool serialize( const node_type & src, const std::string & destfile )
                        {
				if( destfile.empty() ) return false;
                                std::unique_ptr<std::ostream> os( ::s11n::io::get_ostream( destfile ) );
                                if( ! os.get() ) return false;
                                bool b = this->serialize( src, *os );
                                return b;
                        }

                        /**
                           Must be implemented to parse a node_type from the given istream.

                           It should return true on success, false on error.

                           The default implementation always returns 0 and does nothing.

                           Note that this function does not use
                           s11n::deserialize() in any way, and is only
                           coincidentally related to it.

			   Subclasses should try not to have to buffer
			   the whole stream before parsing, because
			   object trees can be arbitrarily large and a
			   buffered copy effectively doubles the
			   memory needed to store the tree during the
			   deserialization process. Buffering
			   behaviour is unspecified by this interface,
			   however, and subclasses may pre-buffer the
			   whole stream content if they need to.
                        */
                        virtual node_type * deserialize( std::istream & )
                        {
                                return 0;
                        }


                        /**
                           Overloaded to load dest from the given filename.

                           It supports zlib/bz2lib decompression for
                           files if your s11n lib supports them.

                           This is virtual for the same reason as
                           serialize(string).

                        */
                        virtual node_type * deserialize( const std::string & src )
                        {
                                typedef std::unique_ptr<std::istream> AP;
                                AP is = AP( ::s11n::io::get_istream( src ) );
                                if( ! is.get() ) return 0;
                                return this->deserialize( *is );
                        }


                        /**
                           Gets this object's magic cookie.

                           Cookies are registered with
                           <code>class_loader< data_node_serializer<NodeType> ></code>
                           types to map files to file input parsers.
                        */
                        std::string magic_cookie() const
                        {
                                return this->m_cookie;
                        }


#if s11n_SERIALIZER_ENABLE_INTERACTIVE
			bool is_cancelled() const { return m_cancelled; }
			void cancel() { this->m_cancelled = true; }

			node_type * deserialize( std::string const & src, progress_reporter & p )
			{
				this->m_prog = &p;
				node_type * n = 0;
				try
				{
					n = this->deserialize( src );
					this->m_prog = 0;
				}
				catch(...)
				{
					this->m_prog = 0;
					throw;
				}
				return n;
			}

			node_type * deserialize( std::istream & src, progress_reporter & p )
			{
				this->m_prog = &p;
				node_type * n = 0;
				try
				{
					n = this->deserialize( src );
					this->m_prog = 0;
				}
				catch(...)
				{
					this->m_prog = 0;
					throw;
				}
				return n;
			}

                        bool serialize( const node_type & src, std::ostream & dest, progress_reporter & p )
                        {
				this->m_prog = &p;
				bool b = false;
				try
				{
					b = this->serialize( src, dest );
					this->m_prog = 0;
				}
				catch(...)
				{
					this->m_prog = 0;
					throw;
				}
				return b;
                        }

                        bool serialize( const node_type & src, std::string const & dest, progress_reporter & p )
                        {
				this->m_prog = &p;
				bool b = false;
				try
				{
					b = this->serialize( src, dest );
					this->m_prog = 0;
				}
				catch(...)
				{
					this->m_prog = 0;
					throw;
				}
				return b;
                        }
#endif // s11n_SERIALIZER_ENABLE_INTERACTIVE

                protected:
                        /**
                           Sets the magic cookie for this type.
                        */
                        void magic_cookie( const std::string & c )
                        {
                                this->m_cookie = c;
                        }

                        /**
                           metadata is an experimental feature
                           allowing serializers to store arbitrary
                           serializer-specific information in their
                           data steams.
                         */
                        node_type & metadata()
                        { return this->m_meta; }
                        /**
                           A const overload of metadata().
                         */
                        const node_type & metadata() const
                        { return this->m_meta;}

#if s11n_SERIALIZER_ENABLE_INTERACTIVE
			void progress( size_t pos, size_t total )
			{
				if( this->m_prog )
				{
					this->m_prog->operator()( pos, total );
				}
			}
			void clear_cancel() { this->m_cancelled = false; }
			void assert_not_cancelled()
			{
				if( this->is_cancelled() )
				{
					throw ::s11n::s11n_exception("Serializer operation was cancelled.");
				}
			}
#endif // s11n_SERIALIZER_ENABLE_INTERACTIVE

                private:
                        std::string m_cookie;
                        node_type m_meta;
#if s11n_SERIALIZER_ENABLE_INTERACTIVE
			bool m_cancelled;
			progress_reporter * m_prog;
#endif
                }; // data_node_serializer<>

		/**
		   Tries to guess which Serializer can be used to read
		   is. Returns an instance of that type on success or
		   0 on error. The caller owns the returned object. It
		   may propagate exceptions.

		   Achtung: the first line of input from the input
		   stream is consumed by this function (to find the
		   cookie), which means that if this stream is handed
		   off to the object which this function returns, it
		   won't get its own cookie.  The only reliable way
		   around this [that i know of] is to buffer the whole
		   input as a string, and i don't wanna do that (it's
		   really bad for massive data files).

		   Special feature:
		   
		   If the first line of the stream is
		   "#s11n::io::serializer CLASSNAME" or
		   "#!/s11n/io/serialize CLASSNAME" then the CLASSNAME
		   token is expected to be a Serializer class
		   name. This function will try to classload that
		   object.  If successful it will use that type to
		   deserialize the input stream. If that fails, it
		   will return 0.  The intention of this feature is to
		   simplify creation of non-C++ tools which generate
		   s11n data (e.g., perl scripts), so that they don't
		   need to know the exact cookies.

		   Added in 1.2.1.
		*/
		template <typename NodeType>
		data_node_serializer<NodeType> * guess_serializer( std::istream & is )
		{
			typedef data_node_serializer<NodeType> ST;
			ST * ser = 0;
                        std::string cookie;
                        // CERR << "cookie="<<cookie<<std::endl;
			cookie = get_magic_cookie( is );
			if( cookie.empty() ) return 0;
			std::string opencmd = "#s11n::io::serializer ";
			std::string::size_type at = cookie.find( opencmd );
			if( std::string::npos == at )
			{ // try new approach, added in 1.1.0:
				opencmd = "#!/s11n/io/serializer ";
				at = cookie.find( opencmd );
			}

			if( 0 == at )
			{
				std::string dll = cookie.substr( opencmd.size() );
				ser = ::s11n::cl::classload<ST>( dll );
			}
			else
			{
				ser =  ::s11n::cl::classload<ST>( cookie );
			}
			return ser;
		}

		/**
		   An overload which assumes infile is a local file.

		   Added in 1.2.1.
		*/
		template <typename NodeType>
		data_node_serializer<NodeType> * guess_serializer( std::string const & infile )
		{
			std::unique_ptr<std::istream> is( get_istream( infile.c_str() ) );
			return is.get()
				? guess_serializer<NodeType>( *is )
				: 0;
		}

                /**

                Tries to load a NodeType object from the given
                node. It uses the cookie from the input stream (the
                first line) and uses
                s11n::cl::classload<SerializerBaseType>() to find a
                matching Serializer.

                On error 0 is returned or an exception is thrown,
                else a new pointer, which the caller owns.

                ACHTUNG: Only usable for loading ROOT nodes.

		See guess_serializer( std::istream & is ) for more
		information, as that function is used to dispatch the
		stream.

                */
                template <typename NodeType>
                NodeType *
                load_node_classload_serializer( std::istream & is )
                {
			try
			{
				typedef data_node_serializer<NodeType> ST;
				std::unique_ptr<ST> ser( guess_serializer<NodeType>( is ) );
				return ser.get()
					? ser->deserialize( is )
					: 0;
			}
			catch( const s11n_exception & sex )
			{
				throw sex;
			}
			catch( const std::exception & ex ) // todo: consider allowing ser->deserialize() to pass through exceptions
			{
				throw ::s11n::io_exception( "%s:%d: forwarded exception: %s", 
							    __FILE__, __LINE__, ex.what() );
			}
			catch( ... )
			{
				throw ::s11n::io_exception( "%s:%d: Stream-level deserialization failed for unknown reason.",
							    __FILE__, __LINE__ );
			}
			return 0;
                }

		/**
		   Overloaded to take a filename. This is handled
		   separately from the stream overload because some
		   Serializers must behave differently in the face of
		   streams. e.g., db-based Serializers typically can't
		   deal with streams.

		   Added in 1.2.1.
		*/
                template <typename NodeType>
                NodeType *
                load_node_classload_serializer( std::string const & src )
                {
			try
			{
				typedef data_node_serializer<NodeType> ST;
				std::unique_ptr<ST> ser( guess_serializer<NodeType>( src ) );
				return ser.get()
					? ser->deserialize( src )
					: 0;
			}
			catch( const s11n_exception & sex )
			{
				throw sex;
			}
			catch( const std::exception & ex ) // todo: consider allowing ser->deserialize() to pass through exceptions
			{
				throw ::s11n::io_exception( "%s:%d: forwarded exception: %s", 
							    __FILE__, __LINE__, ex.what() );
			}
			catch( ... )
			{
				throw ::s11n::io_exception( "%s:%d: Stream-level deserialization failed for unknown reason.",
							    __FILE__, __LINE__ );
			}
			return 0;
                }


                /**
                   Returns a node pointer, parsed from the given stream, using
                   <code>s11n::io::data_node_serializer<NodeType></code>
                   as the base type for looking up a stream handler.

                   ACHTUNG: Only usable for loading ROOT nodes.
                */
                template <typename NodeType>
                NodeType * load_node( std::istream & is )
                {
                        return load_node_classload_serializer< NodeType >( is );
                }

                /**
                   Overloaded form of load_node( istream ), provided for
                   convenience.

                   If ExternalData is true, input is treated as a file,
                   otherwise it is treated as a string containing input
                   to parse.

                   ACHTUNG: Only usable for loading ROOT nodes.

		   Behaviour change in 1.2.1:

		   If (ExternalData) then this call is eventually
		   passed to ASerializer->deserialize(src). In
		   previous versions, src was "converted" to a stream
		   and passed to ASerializer->deserialize(istream),
		   which does not work for some Serializers. This was
		   fixed in 1.2.1 to allow the sqlite3 add-on to play
		   along more transparently with s11nconvert and s11nbrowser.
                */
                template <typename NodeType>
                NodeType * load_node( const std::string & src, bool ExternalData = true )
                {
			if( ! ExternalData )
			{
				typedef std::unique_ptr<std::istream> AP;
				AP is( ::s11n::io::get_istream( src, ExternalData ) );
				if( ! is.get() ) return 0;
				return load_node<NodeType>( *is );
			}
			return load_node_classload_serializer<NodeType>( src );
                }

                /**
                   Tries to load a SerializableT from the given stream.
                   On success returns a new object, else 0.

                   The caller owns the returned pointer.

                   ACHTUNG: Only usable for loading ROOT nodes.
                */
                template <typename NodeT,typename SerializableT>
                SerializableT * load_serializable( std::istream & src )
                {
                        typedef std::unique_ptr<NodeT> AP;
                        AP node( load_node<NodeT>( src ) );
                        if( ! node.get() )
                        {
                                CERR << "load_serializable<>(istream) Could not load a root node from the input.\n";
                                return 0;
                        }
                        return ::s11n::deserialize<NodeT,SerializableT>( *node );
                }

                /**
                   An overloaded form which takes an input string. If
                   ExternalData is true the string is treated as a file
                   name, otherwise it is processed as an input stream.

                   ACHTUNG: Only usable for loading ROOT nodes.

		   Behaviour chagne in 1.2.1 when (ExternalData):
		   load_node(string) is used to load the snode tree,
		   as opposed to load_node(stream). This change was to
		   allow non-stream-friendly Serializers (e.g.,
		   DB-based) to integrate more fully into s11n.
                */
                template <typename NodeT,typename SerializableT>
                SerializableT * load_serializable( const std::string & src, bool ExternalData = true )
                {
			if( ! ExternalData )
			{
				typedef std::unique_ptr<std::istream> AP;
				AP is( ::s11n::io::get_istream( src, ExternalData ) );
				if( ! is.get() )
				{
					// CERR << "load_serializable<>(string) Could not load a root node from the input.\n";
					return 0;
				}
				return load_serializable<NodeT,SerializableT>( *is );
			}
                        typedef std::unique_ptr<NodeT> AP;
                        AP node( load_node<NodeT>( src ) );
                        if( ! node.get() )
                        {
                                // CERR << "load_serializable<>(string) Could not load a root node from the input.\n";
                                return 0;
                        }
                        return ::s11n::deserialize<NodeT,SerializableT>( *node );
		}

        } // namespace io

} // namespace s11n

#endif // s11n_DATA_NODE_IO_H_INCLUDED
