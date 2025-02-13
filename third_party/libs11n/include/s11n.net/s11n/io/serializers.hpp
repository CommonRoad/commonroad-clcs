#ifndef s11n_SERIALIZERS_HPP_INCLUDED
#define s11n_SERIALIZERS_HPP_INCLUDED 1

////////////////////////////////////////////////////////////////////////////////
// serializers.hpp: Some utility code for working with
//  s11n::io Serializer types.
//
// License: Public Domain
// Author: stephan@s11n.net
////////////////////////////////////////////////////////////////////////////////

#include <s11n.net/s11n/classload.hpp> // factory fucntions
#include <s11n.net/s11n/io/data_node_io.hpp> // data_node_serializer<> class.
#include <s11n.net/s11n/s11n_config.hpp>

namespace s11n {
        namespace io {

                /**
		   Intended mainly as a convenience for client
		   applications, serializer_list() populates the
		   target list with the names of registered
		   Serializers.

		   ListT must support push_back(std::string).

                   If onlyShortNames is true (the default) then only
                   "simple" names (only alphanumeric or underscore
                   characters) are put into target, not the "full"
                   names of the classes. This is to make the data more
                   useful in the context of client applications as,
                   e.g., a list of choices for users.
		   
		   When onlyShortNames is false then the list may contain
		   some unsightly magic cookie strings.

		   In either case, the list may very well contain
		   different names for the same underlying Serializer,
		   as most are registered with several aliases.

		   Note that only serializers extending from
		   s11n::io::data_node_serializer<NodeT> are returned.

		   The exact values returned by this function may
		   change in the future. e.g., at the moment it does
		   not return the "real" classnames, only registered
		   aliases, but this is arguable behaviour and may
		   change.
                */
                template <typename NodeT, typename ListT>
                void serializer_list( ListT & target, bool onlyShortNames = true )
                {
			typedef ::s11n::io::data_node_serializer<NodeT> BaseSerT;
			/**
			   Maintenance note: i don't encourage the
			   direc use of s11n::fac here, but that is
			   currently the only way to get the list of
			   class names from the factory layer.
			*/
                        typedef ::s11n::fac::factory_mgr< BaseSerT > SerCL;
			typedef typename SerCL::aliaser_type::alias_map_type AMap;
			SerCL & cl = ::s11n::fac::factory< BaseSerT >();
			typename AMap::const_iterator cit = cl.aliases().begin(),
                                 cet = cl.aliases().end();
                         std::string alias;
                         static const std::string nonspecial = 
                                 "_0123456789abcdefghijklmnopqrstuvwqxyzABCDEFGHIJKLMNOPQRSTUVWQXYZ";
                         for( ; cet != cit; ++cit )
                         {
                                 alias = (*cit).first;
                                 if( onlyShortNames )
                                 { // filter out all but "simple" names:
                                         if( std::string::npos !=
                                             alias.find_first_not_of( nonspecial )
                                             )
                                         {
                                                 continue;
                                         }
                                 }
                                 target.push_back( alias );
                         }
                }


                /**
                   Registers a Serializer type with the s11n::io layer. It must:

                   - be templatized on a DataNodeType

                   - subclass
                   s11n::io::data_node_serializer<DataNodeType>

                   - provide a node_type typedef which is the same as
                   DataNodeType

		   - Be a complete type at the time this function is
		   called.

                   i.e., the conventions of all of the Serializers
                   included with libs11n.


                   Registering makes the type available to the
                   data_node_serializer classloader.

                   Arguments:

                   - classname = SerializerT's stringified class name,
                   minus any template parts. e.g. my_serializer.

                   - alias = a "friendly" name for the SerializerT. By
                   convention this is the Serializer's class name
                   stripped of namespace and any trailing
                   "_serializer" part. The alias should, by
                   convention, be suitable to use via, e.g. entry as a
                   command-line argument.

                   SerializeT's magic_cookie() function is called to
                   alias the cookie as an alias for classname. Thus, a
                   SerializerT object is (briefly) instantiated.

                   Node that this function essentially performs the
                   same operations as the reg_serializer.hpp
                   supermacro, and the two should be equivalent
                   (though this seems easier to use).
                */
                template <typename SerializerT>
                void register_serializer( const std::string & classname, const std::string & alias )
                {
                        //CERR << "register_serializer(" << classname << ","<<alias<<")\n";
                        typedef SerializerT ST;
                        typedef typename ST::node_type NT;
                        typedef s11n::io::data_node_serializer<NT> BaseST;
                        ::s11n::cl::classloader_register_subtype< BaseST, ST >( classname );
                        ::s11n::cl::classloader_alias< BaseST >( alias, classname );
                        ::s11n::cl::classloader_alias< BaseST >( ST().magic_cookie(), classname );
			// ^^^ i don't LIKE having to instantiate ST here, but i don't see an
			// easy way around it.
                }

                /**
                   Returns a Serializer object, which must have been registered with
                   the s11n::cl/s11n::fac classloading API, using an interface type
		   of s11n::io::data_node_serializer<NodeT>. e.g., register_serializer()
		   will do the trick.

		   If no serializer for classname is found and
		   classname does not contain the string
		   "_serializer", then then (classname+"_serializer")
		   is tried. This is intended to ease DLL lookups for
		   the conventional abbreviations for the Serializer
		   classes (i.e., my_serializer).

                   The caller owns the returned pointer, which may be 0.
                */
                template <typename NodeT>
                s11n::io::data_node_serializer<NodeT> *
                create_serializer( const std::string & classname )
                {
                        typedef s11n::io::data_node_serializer<NodeT> BaseSerT;
                        BaseSerT * s = 0;
			// todo: consider using try/catch and return 0, to allow for
			// more transparency when swapping out the core's factory layer.
			// Some factories may throw on load errors.
                        if( 0 != ( s = s11n::cl::classload< BaseSerT >( classname ) ) )
                        {
                                return s;
                        }
                        static const char * addon = "_serializer";
                        if( (std::string::npos == classname.find(addon)) ) // try harder!
                        {
                                std::string harder = classname + addon;
                                // CERR << "Trying harder for " << classname << " --> " << harder << "!\n";
                                s = create_serializer<NodeT>( harder );
                        }
                        return s;
                }

        } // namespace io
} // namespace s11n


#endif // s11n_SERIALIZERS_HPP_INCLUDED
