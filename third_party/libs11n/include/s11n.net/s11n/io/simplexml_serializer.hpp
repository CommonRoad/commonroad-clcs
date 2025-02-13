#ifndef simplexml_SERIALIZER_H_INCLUDED
#define simplexml_SERIALIZER_H_INCLUDED 1

////////////////////////////////////////////////////////////////////////
// simplexml_serializer.hpp: a simple XML dialect for the s11n framework
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////

#include <s11n.net/s11n/io/data_node_format.hpp>
#include <s11n.net/s11n/io/strtool.hpp> // translate_entities()

#include <s11n.net/s11n/traits.hpp> // node_traits
#define MAGIC_COOKIE_SIMPLEXML "<!DOCTYPE s11n::simplexml>"

namespace s11n {
	namespace io {


                /***
                    The sharing namespace defines some "sharing contexts"
                    for use with s11n::Detail::phoenix. They are used to
                    provide contexts in which disparate framework components
                    can share data.
                */
                namespace sharing {
                        /**
                           Sharing context used by simplexml_serializer.
                         */
                        struct simplexml_sharing_context {};
                }
                typedef std::map<std::string,std::string> entity_translation_map;

                /**
                   The entity translations map used by simplexml_serializer.
                 */
                entity_translation_map & simplexml_serializer_translations();



// INDENT() is a helper macro for some serializers.
#define INDENT(LEVEL,ECHO) indent = ""; for( size_t i = 0; i < depth + LEVEL; i++ ) { indent += '\t'; if(ECHO) dest << '\t'; }

                /**
                   De/serializes objects from/to a simple XML grammar,
                   with properties stored as XML attibutes and children
                   stored as subnodes.
                */
                template <typename NodeType>
                class simplexml_serializer : public tree_builder_lexer<NodeType,sharing::simplexml_sharing_context>
                {
                public:
                        typedef NodeType node_type;

                        typedef simplexml_serializer<node_type> this_type; // convenience typedef
                        typedef tree_builder_lexer<node_type,sharing::simplexml_sharing_context> parent_type; // convenience typedef

                        simplexml_serializer() : parent_type( "simplexml_data_nodeFlexLexer" ), m_depth(0)
                        {
                                this->magic_cookie( MAGIC_COOKIE_SIMPLEXML );
                        }

                        virtual ~simplexml_serializer() {}

                        /**
                           Reimplemented to return this type's entity
                           translation map.
                         */
                        virtual const entity_translation_map & entity_translations() const
                        {
                                return simplexml_serializer_translations();
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
                                size_t depth = this->m_depth++;
                                if ( 0 == depth )
                                {
                                        dest << this->magic_cookie() << '\n';
                                }


                                std::string nname = NT::name(src);
                                std::string impl = NT::class_name(src);
                                std::string indent;
                                const entity_translation_map & trans = this->entity_translations();

                                std::string ximpl = impl;
                                ::s11n::io::strtool::translate( ximpl, trans, false );

                                INDENT(0,1);
                                dest << "<" << nname << " s11n_class=\"" << ximpl << "\"";

                                std::string propval;
                                std::string key;

                                typedef typename NT::property_map_type::const_iterator PropIT;
				PropIT it = NT::properties(src).begin();
				PropIT et = NT::properties(src).end();
                                if ( it != et )
                                {
                                        for ( ; it != et; ++it )
                                        {
                                                key = (*it).first;
                                                if ( key == std::string("CDATA") )
                                                        continue;	// special treatment later on
                                                propval = (*it).second;
                                                ::s11n::io::strtool::translate_entities( propval, trans, false );
                                                dest << " " << key << "=\"" << propval << "\"";
                                        }
                                }

                                bool use_closer = false; // if false then an element can <close_itself />
                                if ( NT::is_set( src, std::string("CDATA") ) )
                                {
                                        dest << ">";
                                        use_closer = true;
                                        dest << "<![CDATA[" << NT::get( src, "CDATA", std::string("") ) << "]]>";
                                }


                                bool tailindent = false;

				typedef typename NT::child_list_type CLT;
				typedef typename CLT::const_iterator CLIT;
				CLIT cit = NT::children(src).begin();
				CLIT cet = NT::children(src).end();
                                if( cit != cet )
                                {
                                        if( ! use_closer ) dest << '>';
                                        use_closer = true;
                                        tailindent = true;
                                        dest << '\n';
					for( ; cet != cit; ++cit )
					{
						this->serialize_impl( *(*cit), dest );
					}
                                }

                                dest << ( tailindent ? indent : "" );
                                if( use_closer )
                                {
                                        dest << "</" << nname << '>';
                                }
                                else
                                {
                                        dest << " />";
					// dest << "></"<<nname<<'>';
                                }
                                dest << '\n';

                                if( 0 == depth )
                                {
                                        dest.flush();
                                        // if we don't do this then the client is possibly forced to flush() the stream :/
                                }
                                --this->m_depth;
                                return true;
                        }


                private:
                        size_t m_depth;
                };


	} // namespace io
} // namespace s11n
#undef INDENT

#endif // simplexml_SERIALIZER_H_INCLUDED
