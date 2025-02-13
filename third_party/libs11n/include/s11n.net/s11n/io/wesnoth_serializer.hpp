#ifndef wesnoth_SERIALIZER_H_INCLUDED
#define wesnoth_SERIALIZER_H_INCLUDED 1

////////////////////////////////////////////////////////////////////////
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////

#include <s11n.net/s11n/io/data_node_format.hpp>
#include <s11n.net/s11n/traits.hpp> // node_traits
#include <s11n.net/s11n/io/strtool.hpp> // translate_entities()
#define MAGIC_COOKIE_WESNOTH "#s11n::io::wesnoth_serializer"

#define INDENT(LEVEL,ECHO) indent = ""; for( size_t i = 0; i < depth + LEVEL; i++ ) { indent += '\t'; if(ECHO) dest << '\t'; }

namespace s11n { namespace io {
        namespace sharing {
                /**
                   Sharing context used by wesnoth_serializer.
                */
                struct wesnoth_sharing_context {};

        }

        /** convenience typedef */
        typedef std::map<std::string,std::string> entity_translation_map;

        /**
           The entity translations map used by wesnoth_serializer.
        */
        entity_translation_map & wesnoth_serializer_translations();

        /**
           De/serializes objects from/to this class' config-file-like grammar.
	   It gets its name, and most of its grammar, from the Open Source
	   strategy/adventure game "The Battle for Wesnoth" (www.wesnoth.org).
        */
        template <typename NodeType>
        class wesnoth_serializer : public tree_builder_lexer<NodeType,sharing::wesnoth_sharing_context>
        {
        public:
                typedef NodeType node_type;

                typedef wesnoth_serializer<node_type> this_type; // convenience typedef
                typedef tree_builder_lexer<node_type,sharing::wesnoth_sharing_context> parent_type; // convenience typedef

                wesnoth_serializer() : parent_type( "wesnoth_data_nodeFlexLexer" ), m_depth(0)
                {
                        this->magic_cookie( MAGIC_COOKIE_WESNOTH );
                }

                virtual ~wesnoth_serializer() {}

                /**
                   Reimplemented to return this type's entity
                   translation map.
                */
                virtual const entity_translation_map & entity_translations() const
                {
                        return wesnoth_serializer_translations();
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
                        const static char open = '[';
                        const static char close = ']';

                        std::string nname = NT::name(src);
                        std::string impl = NT::class_name(src);
                        dest << open << nname << "=" << impl << close <<"\n";

                        std::string indent;
                        std::string propval;
                        std::string key;

                        INDENT(0,0);
                        typedef typename NT::property_map_type::const_iterator PCIT;
                        PCIT pit = NT::properties(src).begin();
			PCIT pet = NT::properties(src).end();
                        std::string val;
                        const static std::string nonquoted = "_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
                        for( ; pet != pit; ++pit )
                        {
                                dest << indent << (*pit).first << "=";
				val = (*pit).second;
                                ::s11n::io::strtool::translate_entities( val, this->entity_translations(), false );
                                if( std::string::npos != val.find_first_not_of( nonquoted ) )
                                {
                                        dest << "\"" << val << "\"";
                                }
                                else
                                {
                                        dest << val;
                                }
                                dest << "\n";
                        }
			typedef typename NT::child_list_type CLT;
			typedef typename CLT::const_iterator CLIT;
			CLIT cit = NT::children(src).begin();
			CLIT cet = NT::children(src).end();
                        if( cet != cit )
                        {
                                INDENT(1,0);
 				for( ; cet != cit; ++cit )
 				{
					dest << indent;
 					this->serialize_impl( *(*cit), dest );
 				}
                        }
                        INDENT(0,1);
                        dest << open << "/"<< nname << close <<"\n";
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





} } // namespace s11n::io

#undef INDENT
#endif // wesnoth_SERIALIZER_H_INCLUDED
