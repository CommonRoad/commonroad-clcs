#ifndef parens_SERIALIZER_H_INCLUDED
#define parens_SERIALIZER_H_INCLUDED 1

////////////////////////////////////////////////////////////////////////
// data_node_serializers.hpp: some file parsers for the s11n framework
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////

#include <s11n.net/s11n/io/data_node_format.hpp> // base interfaces
#include <s11n.net/s11n/traits.hpp> // node_traits

#define MAGIC_COOKIE_PARENS "(s11n::parens)"
#define INDENT(LEVEL,ECHO) indent = ""; for( size_t i = 0; i < depth + LEVEL; i++ ) { indent += '\t'; if(ECHO) dest << '\t'; }

namespace s11n {
	namespace io {

                namespace sharing {
                        /**
                           Sharing context used by parens_serializer.
                        */
                        struct parens_sharing_context {};
                }


                /**
                   The entity translations map used by parens_serializer.
                 */
                entity_translation_map & parens_serializer_translations();

                /**
                   De/serializes objects from/to a lisp-like grammar.
                */
                template <typename NodeType>
                class parens_serializer : public tree_builder_lexer<NodeType,sharing::parens_sharing_context>
                {
                public:
                        typedef NodeType node_type;

                        typedef parens_serializer<node_type> this_type; // convenience typedef
                        typedef tree_builder_lexer<node_type,sharing::parens_sharing_context> parent_type; // convenience typedef

                        parens_serializer() : parent_type( "parens_data_nodeFlexLexer" ),
                                              m_depth(0)
                        {
                                this->magic_cookie( MAGIC_COOKIE_PARENS );
                        }

                        virtual ~parens_serializer() {}

                        typedef entity_translation_map translation_map;

                        /**
                           Reimplemented to return this type's entity
                           translation map.
                         */
                        virtual const translation_map & entity_translations() const
                        {
                                return parens_serializer_translations();
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
                                if( 0 == depth )
                                {
                                        dest << this->magic_cookie()
//                                              << "\n(* serializer info: "
//                                              << "\n\t" << PARENS_VERSION
//                                              << "\n\tBuilt " << __TIME__ << " on " __DATE__
//                                              << "\n*)"
                                             << "\n";
                                }

                                std::string indent;
                                std::string implclass = NT::class_name(src);

                                // i know this quote check is fairly expensive, but 2 bytes per
                                // object adds up. Consider: 10k objects would take up
                                // 20k bytes just in classname quotes!
                                std::string quote =
                                        (std::string::npos != implclass.find('<'))
                                        ? "\""
                                        : "";
                                dest << NT::name(src) << "=" << this->m_open
                                     << quote << implclass << quote;

                                typename NT::property_map_type::const_iterator beg = NT::properties(src).begin(),
                                        end = NT::properties(src).end();
                                if( end != beg )
                                {
                                        //INDENT(1,0);
                                        std::for_each(beg, end,
                                                      key_value_serializer<node_type>(
                                                              &(this->entity_translations()),
                                                              dest,
                                                              /* indent + */ ' ' + this->m_open ,
                                                              " ",
                                                              this->m_close )
                                                      );
                                }
                                typename NT::child_list_type::const_iterator chbeg = NT::children(src).begin(),
                                        chend = NT::children(src).end();
                                if( chend != chbeg )
                                { // got children?
                                        dest << '\n';
                                        INDENT(1,0);
					for( ; chend != chbeg; ++chbeg )
					{
						dest << indent;
						this->serialize_impl( *(*chbeg), dest );
					}
                                        INDENT(0,1);
                                }

                                dest << this->m_close << '\n';

                                if( 0 == depth )
                                {
                                        dest.flush(); // << std::endl; // else client may be forced to manually flush().
                                }
                                --this->m_depth;
                                return true;
                        }



                private:
                        size_t m_depth;
                        static const std::string m_open;
                        static const std::string m_close;
                };
                template <typename NodeType>
                const std::string parens_serializer<NodeType>::m_open = "(";
                template <typename NodeType>
                const std::string parens_serializer<NodeType>::m_close = ")";

	} // namespace io
} // namespace s11n

#undef PARENS_VERSION
#undef INDENT

#endif // parens_SERIALIZER_H_INCLUDED
