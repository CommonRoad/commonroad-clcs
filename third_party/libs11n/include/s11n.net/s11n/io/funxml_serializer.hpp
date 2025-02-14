#ifndef funxml_SERIALIZER_H_INCLUDED
#define funxml_SERIALIZER_H_INCLUDED 1

////////////////////////////////////////////////////////////////////////
// funxml_serializer.hpp: an XML dialect for the s11n framework
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////


#include <s11n.net/s11n/io/strtool.hpp> // translate_entities()

#include <s11n.net/s11n/traits.hpp> // node_traits
#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // COUT/CERR
#include <s11n.net/s11n/io/data_node_format.hpp>


#define MAGIC_COOKIE_FUNXML "<!DOCTYPE SerialTree>"

namespace s11n {
	namespace io {
                namespace sharing {
                        /**
                           Sharing context used by funxml_serializer.
                         */
                        struct funxml_sharing_context {};

                }
                /** convenience typedef */
                typedef std::map<std::string,std::string> entity_translation_map;


                /**
                   The entity translations map used by funxml_serializer.
                 */
                entity_translation_map & funxml_serializer_translations();



                /**
                   De/serializes objects from/to a simple XML grammar,
                   with all properties and children stored as subnodes.
                */
                template <typename NodeType>
                class funxml_serializer : public tree_builder_lexer<NodeType,sharing::funxml_sharing_context>
                {
                public:
                        typedef NodeType node_type;

                        typedef funxml_serializer<node_type> this_type; // convenience typedef
                        typedef tree_builder_lexer<node_type,sharing::funxml_sharing_context> parent_type; // convenience typedef

                        funxml_serializer() : parent_type( "funxml_data_nodeFlexLexer" ), m_depth(0)
                        {
                                this->magic_cookie( MAGIC_COOKIE_FUNXML );
                        }

                        virtual ~funxml_serializer() {}

                        /**
                           Reimplemented to return this type's entity
                           translation map.
                         */
                        virtual const entity_translation_map & entity_translations() const
                        {
                                return funxml_serializer_translations();
                        }


                        /**
                           Overridden to parse src using this object's grammar rules.
                        */
                        virtual node_type * deserialize( std::istream & src )
                        {
                                return deserialize_lex_forwarder<node_type,sharing::funxml_sharing_context>( "funxml_data_nodeFlexLexer", src );
                                //CERR << "deserialize() returning " << std::hex << ret << "\n";
                        }

                        virtual node_type * deserialize( const std::string & src )
                        { // Forwarding this avoids an odd compiler error sometimes,
                          // where the compiler doesn't see that this function
                          // exists which called from client code. ???
                                return this->parent_type::deserialize( src );
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
                                size_t depth = this->m_depth++;
                                if ( 0 == depth )
                                {
                                        dest << this->magic_cookie() << '\n';
                                }


                                std::string nname = NT::name(src);
                                std::string impl = NT::class_name(src);
                                strtool::translate_entities( impl, this->entity_translations(), false ); // handle class templates

                                std::string indent;


                                dest << "<" << nname << " class=\"" << impl << "\">\n";
                                typename NT::property_map_type::const_iterator pit = NT::properties(src).begin(),
                                        pet = NT::properties(src).end();


                                std::string propval;
                                std::string key;

                                INDENT(1,0);
                                for ( ; pet != pit; ++pit )
                                {
                                        key = ( *pit ).first;
                                        propval = ( *pit ).second;
                                        strtool::translate_entities( propval, this->entity_translations(), false );
                                        dest << indent;
                                        dest << "<" << key << ">";
                                        dest << propval;
                                        dest << "</" << key << ">\n";
                                }
                                INDENT(1,0);

				typedef typename NT::child_list_type CLT;
				typedef typename CLT::const_iterator CLIT;
 				CLIT cit = NT::children(src).begin();
 				CLIT cet = NT::children(src).end();
 				for( ; cet != cit; ++cit )
 				{
					dest << indent;
 					this->serialize_impl( *(*cit), dest );
 				}

                                INDENT(0,1);
                                dest << "</" << nname << ">\n";
                                if( 0 == depth )
                                {
                                        dest.flush();
                                        // if we don't do this then the client is possibly forced to flush() the stream :/
                                }
                                --this->m_depth;
                                return true;
#undef INDENT
                        }

                private:
                        size_t m_depth;
                };

	} // namespace io
} // namespace s11n

#endif // funxml_SERIALIZER_H_INCLUDED
