#ifndef compact_SERIALIZER_H_INCLUDED
#define compact_SERIALIZER_H_INCLUDED 1

////////////////////////////////////////////////////////////////////////
// compact_serializer.hpp: a binary-like serializer for the s11n framework
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////

#include <s11n.net/s11n/io/data_node_format.hpp> // base interfaces
#include <s11n.net/s11n/traits.hpp> // node_traits

#define MAGIC_COOKIE_COMPACT "51191011"
#define INDENT(LEVEL,ECHO) indent = ""; for( size_t i = 0; i < depth + LEVEL; i++ ) { indent += '\t'; if(ECHO) dest << '\t'; }

namespace s11n {
	namespace io {

                namespace sharing {
                        /**
                           Sharing context used by compact_serializer.
                        */
                        struct compact_sharing_context {};
                }

                /**
                   De/serializes objects from/to a compact binary-like grammar.
                */
                template <typename NodeType>
                class compact_serializer : public tree_builder_lexer<NodeType,sharing::compact_sharing_context>
                {
                public:
                        typedef NodeType node_type;

                        typedef compact_serializer<node_type> this_type; // convenience typedef
                        typedef tree_builder_lexer<node_type,sharing::compact_sharing_context> parent_type; // convenience typedef

                        compact_serializer() : parent_type( "compact_data_nodeFlexLexer" ),
                                              m_depth(0)
                        {
                                this->magic_cookie( MAGIC_COOKIE_COMPACT );
                        }

                        virtual ~compact_serializer() {}

                        typedef entity_translation_map translation_map;

//                         /**
// 			   Tries to read a new node from src. The caller owns the returned
// 			   pointer, which may be null.
//                         */
//                         virtual node_type * deserialize( std::istream & is )
//                         {
//                                 return 0;
//                         }


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
			bool serialize_impl( const node_type & src, std::ostream & dest )
                        {
                                typedef ::s11n::node_traits<node_type> node_traits;

                                static const int ctrlwidth = 2;
                                static const int size2b = 2;
                                static const int size4b = 4;
                                static const int cookiewidth = 8;
                                
                                static const unsigned long Magic_Cookie_4B = 0x51191011;
                                static const unsigned long Node_Open = 0xf1;
                                static const unsigned long Node_Close = 0xf0;
                                static const unsigned long Prop_Open =  0xe1;
                                static const unsigned long Data_End = 0x51190000; // not strictly necessary



                                std::string nname = node_traits::name( src );
                                std::string impl = node_traits::class_name( src );
                                std::string::size_type sizE = 0;
                                std::string::size_type ins;
                                // WTF must all this stream manip be done on every fucking insert?
#define OS_INT(C,W) dest.width(W);dest<<std::hex<<std::right<<(unsigned int)(C);
#define INSERT(vaR,WIDTH) sizE = vaR.size(); OS_INT(sizE,WIDTH); \
                for( ins = 0; ins < sizE; ins++ ) {\
			/*OS_INT(vaR[ins],charwidth);*/ \
			dest << (unsigned char) vaR[ins]; \
		};
                                size_t depth = this->m_depth++;
                                
                                if ( 0 == depth )
                                {
                                        dest.setf( std::ios_base::hex );
                                        dest.fill('0');
                                        dest.setf(std::ios_base::right, std::ios_base::adjustfield);
                                        OS_INT(Magic_Cookie_4B,cookiewidth);
                                        dest << '\n';
                                }

                                OS_INT(Node_Open,ctrlwidth);
                                INSERT(nname,size2b);
                                INSERT(impl,size2b);
                                typedef typename node_traits::property_map_type PMT;
				typedef typename PMT::const_iterator CIT;
                                CIT it = node_traits::properties(src).begin();
                                CIT et = node_traits::properties(src).end();
                                std::string propval;
                                std::string propname;
                                for ( ; it != et; ++it )
                                {
                                        OS_INT(Prop_Open,ctrlwidth);
                                        propname = ( *it ).first;
                                        INSERT(propname,size2b);
                                        propval = ( *it ).second;
                                        INSERT(propval,size4b);
                                        //OS_INT(Prop_Close);
                                }

				typedef typename node_traits::child_list_type CLT;
				typedef typename CLT::const_iterator CLIT;
 				CLIT cit = node_traits::children(src).begin();
 				CLIT cet = node_traits::children(src).end();
 				for( ; cet != cit; ++cit )
 				{
 					this->serialize_impl( *(*cit), dest );
 				}

                                OS_INT(Node_Close,ctrlwidth);
                                dest << '\n';
                                if( 0 == depth )
                                {
                                        OS_INT(Data_End,cookiewidth);
                                        dest << std::endl;
                                        // maintenance: dest must be flush()ed or the client may be forced to do it.
                                }
                                --this->m_depth;
                                return true;
                        }
#undef OS_INT
#undef INSERT



                private:
                        size_t m_depth;
                };

	} // namespace io
} // namespace s11n

#undef INDENT

#endif // compact_SERIALIZER_H_INCLUDED
