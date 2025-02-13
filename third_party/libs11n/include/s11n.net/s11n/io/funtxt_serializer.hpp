#ifndef funtxt_SERIALIZER_H_INCLUDED
#define funtxt_SERIALIZER_H_INCLUDED 1

////////////////////////////////////////////////////////////////////////
// data_node_serializers.hpp: some file parsers for the s11n framework
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////

#include <s11n.net/s11n/io/data_node_format.hpp>
#include <s11n.net/s11n/traits.hpp> // node_traits

#define MAGIC_COOKIE_FUNTXT "#SerialTree 1"

#define INDENT(LEVEL,ECHO) indent = ""; for( size_t i = 0; i < depth + LEVEL; i++ ) { indent += '\t'; if(ECHO) dest << '\t'; }

namespace s11n {
	namespace io {
                namespace sharing {
                        /**
                           Sharing context used by funtxt_serializer.
                         */
                        struct funtxt_sharing_context {};

                }

                /**
                   The entity translations map used by funtxt_serializer.
                 */
                entity_translation_map & funtxt_serializer_translations();


                /**
                   De/serializes objects from/to this class' config-file-like grammar.
                */
                template <typename NodeType>
                class funtxt_serializer : public tree_builder_lexer<NodeType,sharing::funtxt_sharing_context>
                {
                public:
                        typedef NodeType node_type;

                        typedef funtxt_serializer<node_type> this_type; // convenience typedef
                        typedef tree_builder_lexer<node_type,sharing::funtxt_sharing_context> parent_type; // convenience typedef

                        funtxt_serializer() : parent_type( "funtxt_data_nodeFlexLexer" ), m_depth(0)
                        {
                                this->magic_cookie( MAGIC_COOKIE_FUNTXT );
                        }

                        virtual ~funtxt_serializer() {}

                        /**
                           Reimplemented to return this type's entity
                           translation map.
                         */
                        virtual const entity_translation_map & entity_translations() const
                        {
                                return funtxt_serializer_translations();
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
                                std::string quote =
                                        (std::string::npos != impl.find('<'))
                                        ? "\""
                                        : "";

                                dest << nname << " class=" << quote << impl << quote <<"\n";
                                INDENT(0,1);
                                dest <<"{\n";
                                std::string propval;
                                std::string key;

                                INDENT(1,0);
                                std::for_each(NT::properties(src).begin(),
                                              NT::properties(src).end(),
                                              key_value_serializer<node_type>(
                                                                              &(this->entity_translations()),
                                                                              dest,
                                                                              indent,
                                                                              " ",
                                                                              "\n" )
                                              );

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
                                dest << "}\n";
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
#endif // funtxt_SERIALIZER_H_INCLUDED
