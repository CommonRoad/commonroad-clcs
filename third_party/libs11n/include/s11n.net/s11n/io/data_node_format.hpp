#ifndef s11n_DATA_NODE_FORMAT_H_INCLUDED
#define s11n_DATA_NODE_FORMAT_H_INCLUDED
////////////////////////////////////////////////////////////////////////////////
// data_node_format.hpp
// Contains some helpers related to parsing/formating data_node-style objects.
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////////////


#include <string>
#include <list>
#include <map>
#include <stdexcept>

#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // COUT/CERR
#include <s11n.net/s11n/phoenix.hpp>
#include <s11n.net/s11n/io/strtool.hpp>
#include <s11n.net/s11n/classload.hpp> // classload()

#include <s11n.net/s11n/algo.hpp> // some utility functors
#include <s11n.net/s11n/serialize.hpp> // data_node_serializer<> and friends

#include <s11n.net/s11n/traits.hpp>
#include <s11n.net/s11n/io/data_node_io.hpp> // default serializer interfaces

////////////////////////////////////////////////////////////////////////////////
// NO DEPS ON data_node.hpp ALLOWED!
////////////////////////////////////////////////////////////////////////////////

#include <s11n.net/s11n/io/FlexLexer.hpp>
// cl_ABSTRACT_BASE(FlexLexer);


namespace s11n {
        namespace io {

                namespace Private {
                        /**
                           A helper to hide FlexLexer subclasses from needing to be
                           included in header files. (FlexLexer's subclassing technique
                           (via macros) makes them impracticle to mix together
                           in the same headers.)

                           May throw std::runtime_error.
                         */
                        int lex_api_hider_yylex( FlexLexer *, std::istream & );

                }

                /**
                   A typedef representing a map of tokens used for
                   "entity translations" by s11n parsers/serializers.
                */
                typedef std::map<std::string,std::string> entity_translation_map;


                /**
                   tree_builder exists mainly so some lex-based code
                   can get access to a non-templated type (so we don't
                   have to hard-code the parsers to a node_type).

                   It provides only the interface needed by the current
                   lex-based parsers, not some ultimately reusable
                   interface.

                   It is not functionally useful by itself - it must
                   be subclassed and all of it's virtual methods must be
                   implemented.
                */
                class tree_builder
                {
                public:
                        tree_builder() : m_autodel(true) {}

                        virtual ~tree_builder() {}

                        /**
                           Starts a new node with the the given class
                           name and node name.

                           Return value indicates success or failure.
                        */
                        virtual bool
                        open_node( const std::string & classname, const std::string & nodename ) = 0;

                        /**
                           Closes the current node.

                           Return value indicates success or failure.
                        */
                        virtual bool
                        close_node() = 0;

                        /**
                           Sets property key to val for the current node.

                           Return value indicates success or failure.
                        */
                        virtual bool
                        add_property( const std::string & key, const std::string & val ) = 0;

                        /**
                           Returns the depth level of the parser,
                           where the root node is 1.
                        */
                        virtual size_t node_depth() const = 0;


                        /**
                           Changes the implementation class name of
                           the current node.
                        */
                        virtual bool change_node_class( const std::string & newclassname ) = 0;

                        /**
                           If auto_delete() is on (the default) then
                           this object should delete it's children
                           when it is destroyed, otherwise it will
                           not. It is up to subclasses to honor this,
                           as this base type does no handling of
                           children.
                        */
                        void auto_delete( bool b )
                        {
                                this->m_autodel = b;
                        }

                        /**
                           This is the getter for auto_delete( bool ).
                        */
                        bool auto_delete() const
                        {
                                return this->m_autodel;
                        }

			/**
			   Should reset the state of this object to
			   it's default. How it sets the auto_delete()
			   flag is implementation-dependent. The default
			   implementation does nothing, as this type has
			   no state other than the auto_delete() flag.
			*/
			virtual void reset()
			{
			}

                private:
                        bool m_autodel;

                };



                /**
                   tree_builder_context is a helper for accessing some
                   template-dependent code from non-template-aware
                   lexer code. It's usage is admitedly a bit strange
                   (indeed, it's whole existance is).

                   This object sets up a "context channel" where
                   a given FlexLexer can, in a thread-safe manner,
                   communicate data back to a data_node_tree_builder<NodeType>
                   without knowing the exact NodeType.

                   For samples see the s11n lexers, under src/serializers/XXX/XXX_serializer.flex.

                   All of it's methods, except for builder(), mirror
                   those of a tree_builder object, so see that class
                   for the API docs. For the "mirrored" functions, the
                   object being proxied is that set via builder(). It
                   is intended that only the lexers associated with
                   this context actually use it's API.

                   Calling the proxied functions when no builder is
                   set has no effect. Calling them with no bind()ed 
                   FlexLexer may be fatal.

                */
                template <typename ContextT>
                class tree_builder_context
                {
                public:
                        /** The context type for this class. */
                        typedef ContextT context_type;
                        /**
                           Sets the current builder object for this context.

                           Pointer ownership does not change by
                           calling this function.

                           This must be carefully marshalled: it must
                           always be set immediately before the
                           matching lexer is used, and unbind(lexer)
                           should be called immediately afterwards to
                           free up the internal marshaling data. Failing
                           to call unbind will mean a resource leak
                           (albeit a small one).

                           Preconditions:

                           - lexer and builder must be valid pointers
                           and must out-live the expected lifetime of
                           this context object, which internally
                           associates these two objects.
                        */
                        static void bind( const FlexLexer * lexer, tree_builder * builder )
                        {
                                lmap()[lexer].builder = builder;
                        }

                        /**
                           Frees up the internal resources used by the
                           marshaling process for the given lexer.
                        */
                        static void unbind( const FlexLexer * lexer )
                        {
                                lmap().erase( lexer );
                        }

                        /**
                           Gets the current builder object for this
                           context, which must have been previously
                           set up via a call to bind(lexer,builder).

                           Ownership of the returned pointer does not
                           change by calling this function.
                        */
                        static tree_builder * builder( const FlexLexer * lexer )
                        {
                                return lmap()[lexer].builder;
                        }


#define IFNOLEXER(RET) if( lmap().end() == lmap().find(lexer) ) return RET;
                        /**
                           See tree_builder::open_node().
                        */
                        static bool open_node( const FlexLexer * lexer,
                                               const std::string & classname,
                                               const std::string & nodename )
                        {
                                IFNOLEXER(false);
                                return lmap()[lexer].builder->open_node( classname, nodename );
                        }

                        /**
                           See tree_builder::clode_node().
                        */
                        static bool close_node( const FlexLexer * lexer )
                        {
                                IFNOLEXER(false);
                                return lmap()[lexer].builder->close_node();
                        }

                        /**
                           See tree_builder::add_property().
                        */
                        static bool
                        add_property(  const FlexLexer * lexer,
                                       const std::string & key,
                                       const std::string & val )
                        {
                                IFNOLEXER(false);
                                return lmap()[lexer].builder->add_property( key, val );
                        }

                        /**
                           See tree_builder::node_depth().
                        */
                        static size_t node_depth(  const FlexLexer * lexer )
                        {
                                IFNOLEXER(0);
                                return lmap()[lexer].builder->node_depth();
                        }

                        /**
                           See tree_builder::change_node_class().
                        */
                        static bool change_node_class(  const FlexLexer * lexer,
                                                        const std::string & newclassname )
                        {
                                IFNOLEXER(false);
                                return lmap()[lexer].builder->change_node_class( newclassname );
                        }
#undef IFNOLEXER

                        /**
                           This is intended for direct access by a lexer associated
                           with this context, and ONLY by such lexers.

                           Except for the builder member, these are
                           temporary holding points for vars common to
                           most lexers, placed here to avoid using
                           global data in the lexer code.
                        */
                        struct lexer_metadata
                        {
                                tree_builder * builder;

                                size_t internaldepth; // current internal depth (not always the same as node_depth())
                                std::string nodename; // name of current node
                                std::string nodeclass; // class name of current node
                                std::string property; // property value buffer
                                std::string bufferyy; // lexer-dependent
                                lexer_metadata()
                                {
                                        builder = 0;
                                        internaldepth = 0;
                                        nodename = nodeclass = property = bufferyy = "";
                                }
                        };

                        /**
                           Returns the lexer_metadata for the given lexer, creating one
                           if needed. It is assumed that the lexer has been bound via a
                           call to bind().
                        */
                        static lexer_metadata & metadata( const FlexLexer * lexer )
                        {
                                return lmap()[lexer];
                        }

                private:
                        /** Convenience typedef. */
                        typedef tree_builder_context<context_type> this_type;
                        /** lexer-to-metadata map */
                        typedef std::map<const FlexLexer *,lexer_metadata> lexer_map;
                        static lexer_map & lmap()
                        {
                                return ::s11n::Detail::phoenix<
                                        lexer_map,
                                        this_type
                                        >::instance();
                        }

                };

                /**
                   data_node_tree_builder is a helper class for
                   building trees from deserialized data, designed
                   particularly for use with lex/callback-based tree
                   builders.

                   It owns all objects which build up it's tree. If
                   you want them you must manually remove them from the
                   container. You normally do not want them, however - they're
                   mostly throwaway nodes on their way to becoming fully
                   deserialized objects.

                   This class only provides methods for building a tree, not
                   for traversing it. Once you have built a tree, traverse it
                   starting at the root_node().

                   Based on usage conventions this type supports only
                   a single root node.
                */
                template <typename NodeType>
                class data_node_tree_builder : public tree_builder
                {
                public:
                        typedef NodeType node_type;

                        typedef std::list< node_type * > child_list_type;

                        /** Creates a default builder. */
                        data_node_tree_builder() : m_node_count(0), m_node(0),m_root(0)
                        {
                        }

                        /**
                           Deletes this object's children if
                           auto_delete() returns true.
                        */
                        virtual ~data_node_tree_builder()
                        {
                                if( this->auto_delete() && this->m_root )
                                {
                                        //CERR << "data_node_tree_builder<> cleaning up root node.\n";
                                        delete( this->m_root );
                                }
                                else
                                {
                                        //CERR << "data_node_tree_builder<> was relieved of child duty (or had no child).\n";
                                }
                        }


                        /**
                           Opens a new node, making that the current node.
                           classname will be used for the node's impl_class()
                           (see docs for node_type::impl_class()). name will
                           be the object's name, which is important for
                           de/serializing the node (see node_type::name()).

                           It returns false on error, else true. The default
                           implementation has no error conditions, and
                           therefor always returns true.

                           Node that classnames and node names need not be
                           unique (nor make up unique combinations). Any
                           number of nodes may have the same name or
                           classname.
                        */
                        bool open_node( const std::string & classname, const std::string & nodename )
                        {
                                ++m_node_count;

                                this->m_node = ( this->m_nodestack.empty() 
                                                 ? 0
                                                 : this->m_nodestack.back() );
                                typedef ::s11n::node_traits<node_type> NTR;
                                node_type * newnode = new node_type();
                                if ( m_node )
                                { // if we're in a node, add new node as a child to that one:
                                        NTR::children( *m_node ).push_back( newnode );
                                }
                                this->m_node = newnode;
                                NTR::name( *m_node, nodename );
                                NTR::class_name( *m_node, classname );
                                this->m_nodestack.push_back( m_node );
                                bool ret = true;
                                if ( 1 == this->m_nodestack.size() )
                                {
                                        if( m_root )
                                        {
                                                CERR << "open_node("<<classname<<","<<nodename<<") WARNING: deleting extra root node!\n";
                                                delete( m_node );
						m_node = 0;
                                                ret = false;
                                        }
                                        else
                                        {
                                                m_root = m_node;
                                        }
                                }
                                return ret;
                        }

                        /**
                           Closes the most-recently-opened node, effectively
                           popping the previous node off of the node stack (it
                           is not destroyed).  It is an error to call this more
                           often than calling open_node().

                           It returns false on error (e.g., called
                           with no opened node).
                        */
                        virtual bool close_node()
                        {
                                if ( !m_node || m_nodestack.empty() )
                                {
                                        CERR << "close_node() error: called with an empty node stack!" << std::endl;
                                        return false;
                                }
                                m_nodestack.pop_back();
                                if ( m_nodestack.empty() )
                                {
                                        m_node = NULL;
                                }
                                else
                                {
                                        m_node = m_nodestack.back();
                                }
                                return true;
                        }


                        /**
                           Adds the given key/value pair to the
                           current node and returns true. If no node
                           is currently opened it returns false.
                        */
                        virtual bool add_property( const std::string & key, const std::string & val )
                        {
                                if( ! this->m_node ) return false;
                                typedef ::s11n::node_traits<node_type> NTR;
                                NTR::set( *m_node, key, val );
                                return true;
                        }

                        /**
                           Returns the total number of nodes opened via open_node().
                        */
                        size_t node_count() const
                        {
                                return m_node_count;
                        }


                        /**
                           Returns the current depth of opened nodes. A return
                           value of 1 means the current node is the root node,
                           for example, and 0 means that no node has yet been
                           opened.
                        */
                        size_t node_depth() const
                        {
                                return m_nodestack.size();
                        }


                        /**
                           Returns the most recent root node parsed out of the
                           input object.

                           Use auto_delete() to determine ownership of
                           the returned pointer.
                        */
                        node_type * root_node() const
                        {
                                return m_root;
                        }


                        /**
                           Returns the current node.

                           Use auto_delete() to determine ownership of
                           the returned pointer.
                        */
                        node_type * current_node() const
                        {
                                return m_node;
                        }

                        /**
                           Changes class name of current node, if one
                           is set. Returns false only if no node is
                           currently opened, else it returns true.
                        */
                        virtual bool change_node_class( const std::string & newclassname )
                        {
                                if( ! this->m_node ) return false;
                                typedef ::s11n::node_traits<node_type> NTR;
                                NTR::class_name( *(this->m_node), newclassname );
                                return true;
                        }

                        /**
                           Deletes this object's root_node() if auto_delete() is true. Cleans
			   up all internal data structures so that this object has an empty
			   state.
                        */
                        virtual void reset()
                        {
                                if( this->auto_delete() && this->m_root ) delete this->m_root;
                                this->m_root = 0;
				// todo: back-port the following three lines to 1.0.x:
				this->m_node = 0;
				this->m_node_count = 0;
				this->m_nodestack = node_stack();
                        }

                private:
                        size_t m_node_count;
                        node_type * m_node;
                        node_type * m_root;
                        typedef std::deque < node_type * > node_stack;
                        node_stack m_nodestack;                        
                };



                /**
                   This function exists for a really long, strange
                   reason involving accessing templatized types from
                   template-free code (FlexLexers).

                   - lexerClassName is the name of a FlexLexer
                   subclass. It must be registered with the FlexLexer
                   classloader.

                   - src is the stream to pass on to the lexer.

                   - BuilderContext should be the same one expected by
                   the specific lexer. See the existing lexers for
                   examples. You want to pass the actual
                   BuilderContext's context here, not a
                   tree_builder_context<> type.

                   The caller owns the returned poiner, which may be 0.

		   As of s11n version 1.1.3, this function may throw
		   on error, and is guaranteed to propagate any
		   exceptions it catches (after cleaning up). Except
		   when passing on a (...) exception, all thrown
		   exceptions are guaranteed to be std::exception
		   types.
                */
                template <typename NodeType, typename BuilderContext>
                NodeType * deserialize_lex_forwarder( const std::string & lexerClassName,
                                                      std::istream & src
                                                      )
                {
                        // CERR << "deserialize_lex_forwarder("<<lexerClassName<<")\n";
			std::unique_ptr<FlexLexer> lexer( ::s11n::cl::classload<FlexLexer>( lexerClassName ) );
                        if( ! lexer.get() )
                        {
				throw ::s11n::s11n_exception(
							     "%s:%d: s11n::io::deserialize_lex_forwarder(): Lexer '%s' was not found by classload<FlexLexer>(). It is probably not registered with the classloader.",
							     __FILE__,
							     __LINE__,
							     lexerClassName.c_str() );
                        }

                        typedef s11n::io::data_node_tree_builder<NodeType> BuilderType;
                        typedef tree_builder_context<BuilderContext> BC;
                        std::unique_ptr<BuilderType> treebuilder( new BuilderType );
                        treebuilder->auto_delete( true ); // if we throw, let it clean up
                        try
                        {
                                BC::bind( lexer.get(), treebuilder.get() );
                                // ^^^ sets up the comm channel between the builder and lexer
                                Private::lex_api_hider_yylex(lexer.get(),src); // executes the lexer
                        }
                        catch ( const std::exception & ex )
                        {
				BC::unbind( lexer.get() ); // free up lexer-to-builder binding
				throw ex;
                        }
                        catch (...)
                        {
				BC::unbind( lexer.get() ); // free up lexer-to-builder binding
                                throw;
                        }
                        BC::unbind( lexer.get() ); // free up lexer-to-builder binding
			treebuilder->auto_delete( false ); // we're taking ownership of the children
			return treebuilder->root_node();
                }

                /**
                   tree_builder_lexer is a type intended to ease the
                   implementation of lex-based node tree parsers.

                   It is useless standalone: it must be subclassed.

                   It holds the class name of a FlexLexer type so it
                   can be dynamically loaded as needed. It takes the
                   responsibility of instantiating that type and
                   passing off input to subclasses.
                */
                template <typename NodeType, typename LexerSharingContext>
                class tree_builder_lexer : public data_node_serializer<NodeType>
                {
                public:

                        typedef NodeType node_type;
                        typedef LexerSharingContext sharing_context;
                        typedef data_node_serializer<NodeType> parent_type; // convenience typedef.

                        /**
                           lexerClassName = the class name of the FlexLexer subtype
                           associated with this serializer.
                        */
                        explicit tree_builder_lexer( const std::string & lexerClassName )
                                : m_impl(lexerClassName)
                        {}

                        virtual ~tree_builder_lexer(){}

                        /**
                           Overridden to parse src using this object's lexer.
                           It uses <code>deserialize_lex_forwarder<sharing_context>()</code>,
                           passing it this object's lexer_class().
                        */
                        virtual node_type * deserialize( std::istream & src )
                        {
                                return deserialize_lex_forwarder<
                                        node_type,
                                        sharing_context
                                        >( this->lexer_class(), src );
                        }

                        virtual node_type * deserialize( const std::string & src )
                        { // Forwarding this avoids an odd compiler error sometimes,
                          // where the compiler doesn't see that this function
                          // exists which called from client code. ???
                                return this->parent_type::deserialize( src );
                        }


                        /**
                           Returns this object's lexer class name.
                        */
                        std::string lexer_class() const { return this->m_impl; }


                protected:
                        /**
                           Sets this object's lexer class name.
                        */
                        void lexer_class( const std::string & classname )
                        {
                                this->m_impl = classname;
                        }

                private:
                        std::string m_impl; // implementation class name for a FlexLexer subclass
                };




                /**
                   A helper for serializing properties to a
                   stream. Intended for use by Serializers, not
                   Serializables.

                   NodeType is the container type used for data
                   serialization (e.g. s11n::data_node).

                */
                template <typename NodeType>
                class key_value_serializer
                {
                public:
                        typedef NodeType node_type;
                        typedef typename node_type::value_type pair_type;

                        typedef std::map<std::string,std::string> entity_translation_map;

                        /**
                           map: needed so we can do entity translation
                           in a unified manner here. It must outlive
                           this object. Pass it 0 for no translation.
                           Translations are only applied on VALUES,
                           not KEYS.

                           prefix: inserted before each property.

                           separator: inserted between the key and value.

                           suffix: appended after each entry.


                        */
                        key_value_serializer( const entity_translation_map * map,
                                              std::ostream & dest,
                                              const std::string & prefix,
                                              const std::string & separator,
                                              const std::string & suffix
                                              )
                                : m_pre(prefix), m_sep(separator), m_suf(suffix), m_os(dest), m_map(map)
                        {
                        }

                        /**
                           Sends the following formatted string to os:

                           {prefix}{src.first}{separator}{src.second}{suffix}

                        */
                        void operator()( const pair_type & src ) const
                        {
                                static const std::string errval = "";
                                std::string key = strtool::to( src.first );
                                std::string val = strtool::to( src.second );
                                // should we xlate the key as well?
                                if( this->m_map )
                                {
                                       strtool::translate_entities( val, *(this->m_map) );
                                }
                                this->m_os << this->m_pre;
                                this->m_os << key;
                                this->m_os << this->m_sep;
                                this->m_os << val;
                                this->m_os << this->m_suf;
                        }
                private:
                        std::string m_pre;
                        std::string m_sep;
                        std::string m_suf;
                        std::ostream & m_os;
                        const entity_translation_map * m_map;
                };





                /**
                   A helper functor to loop over serializable children
                   of a node from within a Serializer implementation.

                   Designed for use with std::for_each().

                   SerializerT must be compatible with
                   <code>data_node_serializer<></code>.

                */
                template <typename SerializerT>
                struct node_child_simple_formatter
                {
                        typedef SerializerT serializer_type;
//                         typedef typename SerializerT::node_type node_type;
                        /**
                           Preconditions:

                           - Ser must be valid references.

                           - Both ser and os must outlive this
                           object. More correctly, this object's
                           operator() must not be called after either
                           ser or os are destroyed.

                        */
                        node_child_simple_formatter( serializer_type & ser, std::ostream & os,
                                               const std::string & prefix = "", const std::string & suffix = "\n" )
                                : m_ser(ser), m_os(&os), m_pre(prefix), m_suf(suffix)
                        {
                        }

                        /**
                           Serializes src into this object's target
                           container, using this object's serializer.
                        */
                        template <typename NodeType>
                        bool operator()( const NodeType * src ) const
                        {
                                if( ! src ) return false;
                                if( ! this->m_pre.empty() ) *(this->m_os) << this->m_pre;
                                bool b = this->m_ser.serialize( *src, *(this->m_os) );
                                if( ! this->m_suf.empty() ) *(this->m_os) << this->m_suf;
                                return b;
                        }

                private:
                        serializer_type & m_ser;
                        std::ostream * m_os;
                        std::string m_pre;
                        std::string m_suf;
                };



        } // namespace io
} // namespace s11n

// cl_CLASSLOADER_ABSTRACT_BASE(s11n::io::tree_builder);
#include <s11n.net/s11n/factory.hpp>
#define S11N_FACREG_TYPE s11n::io::tree_builder
#define S11N_FACREG_TYPE_NAME "s11n::io::tree_builder"
#define S11N_FACREG_TYPE_IS_ABSTRACT 1
#include <s11n.net/s11n/factory_reg.hpp>

#endif // s11n_DATA_NODE_FORMAT_H_INCLUDED
