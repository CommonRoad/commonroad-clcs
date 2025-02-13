////////////////////////////////////////////////////////////////////////
// s11nlite.cpp: impl file for s11nlite.h
// author: stephan@s11n.net
// license: public domain
////////////////////////////////////////////////////////////////////////
#include <list>
#include <string>
#include <stdlib.h> // ::getenv()
#include <s11n.net/s11n/s11nlite.hpp>
#include <s11n.net/s11n/s11n_config.hpp>
#include <s11n.net/s11n/s11n_debuggering_macros.hpp>
#include <s11n.net/s11n/io/strtool.hpp>
// #include <s11n.net/stringutil/stdstring_tokenizer.hpp>
#include <s11n.net/s11n/io/serializers.hpp> // create_serializer()
#include <memory> // auto_ptr<>

#define S11NLITE_CONFIG_FILE std::string("${HOME}/.s11nlite-1.1.conf")
#define S11NLITE_SERIALIZER_KEY std::string("serializer_class")
namespace s11nlite {

	static client_interface * m_inst = 0;

	client_interface & instance()
	{
		return m_inst ? *m_inst : client_interface::instance();
	}

	void instance( client_interface * newinst )
	{
		m_inst = newinst;
		return;
	}


	/**
	   INTERNAL.

	   Returns a reference to s11nlite's configuration object. It
	   is read from $HOME/.s11nlite at library startup and saved
	   at library shutdown. s11n clients are encouraged to use their
	   own configuration storage, but this one "may" be used by
	   s11n-related clients (like s11nbrowser).
         */
         node_type & config();


        /**
           Class name of current Serializer.
        */
// 	std::string m_serializer = s11n_S11NLITE_DEFAULT_SERIALIZER_TYPE_NAME;
        std::string m_configfile;

        node_type * m_confignode = 0;

        struct config_saver
        {
                // when this is dtor'd config() will be saved.
                ~config_saver()
                {
                        //CERR << "Saving s11nlite config file ["<<m_configfile<<"]...\n";
                        save( *m_confignode, m_configfile );
                        delete m_confignode;
                        m_confignode = 0;
                }
        };


//         int config_init = (config(),0);

        node_type & config()
        {
                static config_saver m_confsaver;
                if( ! m_confignode )
                {
                        m_confignode = node_traits::create( "s11nlite_config" );
			::s11n::io::strtool::entity_map emap;
			const char * hm = ::getenv("HOME");
			emap["HOME"] = hm ? hm : "/etc";
                        m_configfile = ::s11n::io::strtool::expand_dollar_refs( S11NLITE_CONFIG_FILE, emap );
                        //CERR << "Loading s11nlite config file ["<<m_configfile<<"]\n";
                        node_type * tmp = load_node( m_configfile );
                        if( tmp )
                        {
                                *m_confignode = *tmp;
                                delete tmp;
                        }
                        else
                        {
                                CERR << "s11nlite config file ["<<m_configfile<<"] not found or loading failed. Creating it...\n";
                                if( ! save( *m_confignode, m_configfile ) )
                                {
                                        CERR << "Could not create ["<<m_configfile<<"]! You may want to create one to avoid these silly error messages!\n";
                                        return *m_confignode;
                                }
                        }
                }
                return *m_confignode;
        }


        void serializer_class( const std::string & c )
        {
		instance().serializer_class( c );
        }

        std::string serializer_class()
        {
		std::string sc = instance().serializer_class();
		return sc.empty() ? s11n_S11NLITE_DEFAULT_SERIALIZER_TYPE_NAME : sc;
        }

        serializer_interface *
        create_serializer( const std::string & classname )
        {
                return ::s11n::io::create_serializer<node_type>( classname );
        }

        serializer_interface * create_serializer()
        {
                return create_serializer( serializer_class() );
        }


        node_type *
        find_child( node_type & parent,
                    const std::string subnodename )
        {
                return s11n::find_child_by_name( parent, subnodename );
        }

        const node_type *
        find_child( const node_type & parent,
                    const std::string subnodename )
        {
                return s11n::find_child_by_name( parent, subnodename );
        }

        bool save( const node_type & src, std::ostream & dest )
        {
		return instance().save( src, dest );
        }


        bool save( const node_type & src, const std::string & dest )
        {
		return instance().save( src, dest );
        }

        node_type * load_node( const std::string & src )
        {
		return instance().load_node( src );
        }

        node_type * load_node( std::istream & src )
        {
		return instance().load_node( src );
        }

        bool load_node( const std::string & src, node_type & dest )
	{
	    typedef std::unique_ptr<node_type> AP;
	    AP n( load_node( src ) );
	    if( ! n.get() ) return false;
	    node_traits::swap( dest, *n );
	    return true;
	}

        bool load_node( std::istream & src, node_type & dest )
	{
	    typedef std::unique_ptr<node_type> AP;
	    AP n( load_node( src ) );
	    if( ! n.get() ) return false;
	    node_traits::swap( dest, *n );
	    return true;
	}


} // namespace

#undef S11NLITE_CONFIG_FILE
