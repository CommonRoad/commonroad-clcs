#ifndef s11n_EXPAT_SERIALIZER_HPP_INCLUDED
#define s11n_EXPAT_SERIALIZER_HPP_INCLUDED 1

#include <s11n.net/s11n/io/strtool.hpp> // translate()
#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // COUT/CERR

#include <s11n.net/s11n/io/data_node_format.hpp>

#include <s11n.net/s11n/traits.hpp> // node_traits

#include <expat.h>



#define MAGIC_COOKIE_EXPAT_XML "<!DOCTYPE s11n::io::expat_serializer>"

#include <stdexcept>
#include <sstream>

#define EXPATDEBUG if(0) CERR
#define EXPAT_CLASS_ATTRIBUTE "class"
namespace s11n {

        namespace io {

                namespace sharing {
                        /**
                           Sharing context used by expat_serializer.
                         */
                        struct expat_sharing_context {};

                }
                /** convenience typedef */
                typedef std::map<std::string,std::string> entity_translation_map;


                /**
                   The entity translations map used by expat_serializer.
                 */
                entity_translation_map & expat_serializer_translations();
                /**
                   expat_serializer is an XML-based Serializer, using
                   libexpat to read it's data.
                */
                template <typename NodeType>
                class expat_serializer : public data_node_serializer<NodeType>
                {
                public:
                        typedef NodeType node_type;

			/**
			   Performs a "fast" check for XML key validity
			   on s: if the first char is alpha or underscore,
			   the function returns true, else it returns false.
			*/
			static bool is_valid_xml_key( const std::string & s )
			{
				return (
					(!s.empty())
					&&
					(
					 std::isalpha(s[0])
					 ||
					 ('_' == s[0])
					 )
					);
			}

                        typedef expat_serializer<node_type> this_type; // convenience typedef

                        expat_serializer() : m_depth(0)
                        {
                                this->magic_cookie( MAGIC_COOKIE_EXPAT_XML );
                        }

                        virtual ~expat_serializer() {}

                        /**
                           Overridden to parse src using this object's grammar rules.
                        */
                        virtual node_type * deserialize( std::istream & src )
                        {
                                return this->expat_parse_stream( src );
                        }


                        /**
                           Writes src out to dest.
                        */
                        virtual bool serialize( const node_type & src, std::ostream & dest )
			{
				try
				{
					return this->serialize_impl( src, dest );
				}
				catch(...)
				{
					this->m_depth = 0;
					throw;
				}
				return false; // can't get this far.
			}

		private:

                        /**
                           Writes src out to dest.
                        */
                        bool serialize_impl( const node_type & src, std::ostream & dest )
                        {
                                typedef ::s11n::node_traits<node_type> NT;

// INDENT() is a helper macro for some serializers.
#define INDENT(LEVEL,ECHO) indent = ""; for( size_t i = 0; i < depth + LEVEL; i++ ) { indent += '\t'; if(ECHO) dest << '\t'; }

                                //std::ostringstream buff; // we buffer so we can handle self-closing nodes
                                size_t depth = this->m_depth++;
                                if ( 0 == depth )
                                {
                                        dest << this->magic_cookie() << '\n';
                                }


                                std::string nname = NT::name(src);
				if( ! is_valid_xml_key( nname ) )
				{
					throw s11n::io_exception( "%s:%d: expat_serializer::serialize(): node name '%s' is not a valid XML tag name",
								  __FILE__, __LINE__, nname.c_str() );
				}

                                std::string impl = NT::class_name(src);
                                ::s11n::io::strtool::translate_entities( impl, expat_serializer_translations(), false );
                                std::string indent;
                                dest << "<" << nname << " "<<EXPAT_CLASS_ATTRIBUTE<<"=\""<< impl <<"\"";
                                bool closed = false;
                                typedef typename NT::property_map_type PMT;
				typedef typename PMT::const_iterator CHIT;
				CHIT cit, cet;
				cit = NT::properties(src).begin();
				cet = NT::properties(src).end();
                                std::string propval;
                                std::string key;

                                if( cet != cit )
                                { // got properties?
                                        closed = true;
                                        dest << ">\n";
                                        INDENT(1,0);
                                        for ( ; cet != cit; ++cit )
                                        {
                                                key = ( *cit ).first;
						if( ! is_valid_xml_key( key ) )
						{
							throw s11n::io_exception( "%s:%d: expat_serializer::serialize(): key '%s' is not a valid XML tag name.",
										  __FILE__, __LINE__, key.c_str() );
						}
                                                propval = ( *cit ).second;
                                                dest << indent << "<" << key;
                                                if( propval.empty() )
                                                {
                                                        dest << "/>";
                                                }
                                                else
                                                {
                                                        dest << ">";
                                                        ::s11n::io::strtool::translate_entities( propval, expat_serializer_translations(), false );
                                                        dest << propval;
                                                        dest << "</" << key << ">";
                                                }
                                                dest << "\n";
                                        }
                                }

				
				typedef typename NT::child_list_type CHLT;
                                typename CHLT::const_iterator chit = NT::children(src).begin(),
                                        chet = NT::children(src).end();
                                if( chet != chit )
                                { // got kids?
                                        if( !closed )
                                        { // close node opener
                                                dest << ">\n";
                                                closed = true;
                                        }
                                        INDENT(1,0);
					for( ; chet != chit; ++chit )
					{
						dest << indent;
						this->serialize_impl( *(*chit), dest );
					}
                                }

                                if( closed )
                                {
                                        INDENT(0,1);
                                        dest << "</" << nname << ">";
                                }
                                else // self-close node
                                {
                                        dest << "/>";
                                }

				dest << '\n';
                                if( 0 == depth )
                                {
                                        dest.flush();
                                        // if we don't do this then the client is possibly forced to flush() the stream :/
                                }
                                --this->m_depth;
                                return true;
#undef INDENT
                        }

                        static void XMLCALL start_node( void *, const char * name, const char ** attr )
                        {
                                m_cbuf = "";
                                if( attr[0] )
                                { // object node
                                        std::string clname = "WTF?";
                                        const std::string classnameattr = std::string(EXPAT_CLASS_ATTRIBUTE);
                                        for( int i = 0; attr[i]; i += 2 )
                                        {
                                                if( attr[i] == classnameattr )
                                                {
                                                        clname = attr[i+1];
                                                        break;
                                                }
                                        }
                                        EXPATDEBUG << "Opening object node["<<clname<<","<<name<<"]\n";
                                        m_builder.open_node( clname, name );
                                }
                                else // property node
                                {
                                        m_name = name;
                                        EXPATDEBUG << "Opening property node["<<m_name<<"]\n";
                                }
                        }

                        /**
                           expat end-node callback.
                        */
                        static void XMLCALL end_node( void *, const char * )
                        {
                                if( ! m_name.empty() ) // property node
                                {
                                        EXPATDEBUG << "Closing property node["<<m_name<<"="<<m_cbuf<<"]\n";
                                        m_builder.add_property( m_name, m_cbuf );
                                }
                                else
                                { // object_node
                                        EXPATDEBUG << "Closing object node.\n";
                                        m_builder.close_node();
                                }
                                m_name = "";
                                m_cbuf = "";
                        }

                        /**
                           expat char-data callback.
                        */
                        static void XMLCALL char_handler( void *, const char * txt, int len )
                        {
                                if( m_name.empty() ) return; // we're not in a property.
                                const char *c;
                                for( int i = 0; i < len; i++ )
                                {
                                        c = txt++;
                                        m_cbuf += *c;
                                }
                                EXPATDEBUG << "char_handler(...,"<<len<<"): m_cbuf=[" << m_cbuf << "]\n";
                        }

                        /**
                           Uses expat to try to parse a tree of nodes
                           from the given stream.
                        */
                       node_type * expat_parse_stream( std::istream & is )
                        {
                                XML_Parser p = XML_ParserCreate(NULL);
                                if (! p)
                                {
                                        EXPATDEBUG <<  "Couldn't allocate memory for parser\n";
                                        return 0;
                                }
                                
                                XML_SetElementHandler(p, start_node, end_node);
                                XML_SetCharacterDataHandler( p, char_handler );
                                m_builder.reset();

                                m_name = "";
                                m_cbuf = "";

                                bool done = false;
                                size_t len = 0;
                                std::string buff;
                                try
                                {
                                        while( true )
                                        {
                                                if( std::getline( is, buff ).eof() ) done = true;
                                                len = buff.size();
                                                if( 0 < len ) if (XML_Parse(p, buff.c_str(), len, done) == XML_STATUS_ERROR)
                                                {
                                                        std::ostringstream err;
                                                        err << "Parse error at line "
                                                             << XML_GetCurrentLineNumber(p)
                                                             << ": "
                                                            << XML_ErrorString(XML_GetErrorCode(p))
                                                            << ": buffer=["<<buff<<"]";
//                                                         EXPATDEBUG << err.str() << "\n";
                                                        throw s11n::io_exception( "%s:%d: Parse error at line %d: %s: buffer=[%s]",
										  __FILE__,
										  __LINE__<
										  XML_GetCurrentLineNumber(p),
										  XML_ErrorString(XML_GetErrorCode(p)),
										  buff.c_str() );
                                                }
                                                if( done ) break;
                                        }
                                }
                                catch( std::runtime_error & ex )
                                {
                                        CERR << "EXCEPTION while XML_Parsing input: " << ex.what() << "!\n";
                                        m_builder.auto_delete( true );
                                        m_builder.reset();
                                }
                                XML_ParserFree( p );
                                m_builder.auto_delete( false );
                                return m_builder.root_node();
                        }


                private:
                        size_t m_depth;
                        static data_node_tree_builder<node_type> m_builder;
                        static std::string m_name; // current node's name
                        static std::string m_cbuf; // cdata buffer
                };
                template <typename NodeT> data_node_tree_builder<NodeT> expat_serializer<NodeT>::m_builder;
                template <typename NodeT> std::string expat_serializer<NodeT>::m_name;
                template <typename NodeT> std::string expat_serializer<NodeT>::m_cbuf;

        } // namespace io
} // namespace s11n

#undef EXPATDEBUG
#undef EXPAT_CLASS_ATTRIBUTE
#endif // s11n_EXPAT_SERIALIZER_HPP_INCLUDED
