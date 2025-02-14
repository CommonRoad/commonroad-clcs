#ifndef s11n_ABSTRACT_CREATOR_H_INCLUDED
#define s11n_ABSTRACT_CREATOR_H_INCLUDED 1

////////////////////////////////////////////////////////////////////////
// abstract_creatore.hpp:
// Defines a difficult-to-describe class. See the docs.
//
//    Author: stephan beal <stephan@s11n.net>
//    License: Public Domain
////////////////////////////////////////////////////////////////////////

#include <string>
#include "classload.hpp" // classloader-related funcs
#include <s11n.net/s11n/traits.hpp> // s11n_traits

namespace s11n { namespace Private {


        /**
           abstract_creator is an internal helper to avoid some code
           having to know if a type is created on a stack or the
           heap. This makes some template code easier to write, as it
           avoids syntax errors when trying something like:

<pre>
if( object type is a pointer type ) delete(object);
else { ... }
</pre>

           This implementation creates items the stack via the default
           ctor. If instantiated with (T *) a pointer/heap-based
           specialization is activated instead.

           Designed for use with, e.g., ListType<T>, for which T
           may be either T or (T *).

           These objects contain no state information.

	   The member function release() was removed in s11n 1.1.3,
	   as it was potentially dangerous. s11n_traits::cleanup_functor
	   replaces that functionality and does so correctly.
        */
        template <typename T>
        struct abstract_creator
        {
                /** Same as (T). */
                typedef T value_type;

                /** Same as (T). */
                typedef T base_value_type;

                /**
                   This implementation assigns v to value_type() and
                   returns true.

                   The string argument is bogus for this implementation,
                   and is used by the pointer specialization to implement
                   polymorphic classloading of value_type.

                   Historical note: before s11n release 0.9.17 this function
                   incorrectly did nothing. While this works for many
                   cases, it caused a subtle, hard-to-track bug in
                   s11n::list::deserialize_list(), in which a deserialized
                   list was re-used when the function was called recursively.
                   Many thanks to Patrick Lin for reporting that problem.

                */
                static bool create( value_type & v,
                                    const std::string & /* implclass */  = std::string()  ) throw()
                {
                        // if( 0 ) CERR << "create(&) implclass="<<implclass<<": initializing @" <<std::hex<<&v <<"\n";
			try
			{
				v = value_type();
			}
			catch(...)
			{
				return false;
			}
                        return true;
                }

        }; // abstract_creator<T>

        /**
           A specialization of abstract_creator to create objects on
           the heap, using the s11n::cl API.
        */
        template <typename T>
        struct abstract_creator<T *>
        {
                /** Same as (T *). */
                typedef T * value_type;
 
               /** Same as (T). */
                typedef T base_value_type;

                /**
                   Tries to create a value_type object,
                   using classload&lt;base_value_type&gt;( key )
                   to create it. v is assigned to it's value, which
                   may be 0. Returns true if an object is created,
                   else false.

                   If the call succeeds, the caller owns the memory
                   pointed to by v.

		   If implclass is NOT passed then it must guess and
		   for polymorphic types it is likely to guess
		   incorrectly. It relies on
		   s11n_traits<base_value_type>::class_name(v)
		   returning a valid value, which is normally only
		   useful for monomorphic types and base-most
		   Serializables in a registered hierarchy (which are,
		   for s11n purposes, the same thing).

                   Maintenance note: new classloader registrations may
                   need to be installed as new types show up,
                   especially for streamables/PODs (because those
                   aren't normally registered as classes), or this
                   function won't handle them. In s11n this
                   registration is handled by most of the various
                   proxy installation macros. See the headers under
		   s11n.net/s11n/proxy/pod_*.hpp for examples.
                */
                static bool create(  value_type & v,
                                     const std::string & implclass = std::string() /* guess! */  ) throw()
                {
			try
			{
				typedef s11n_traits<base_value_type> STR;
				const std::string key = 
					(! implclass.empty())
					? implclass
					: STR::class_name(v); // reminder: v is a ref to a pointer!
				v = ::s11n::cl::classload<base_value_type>( key );
				// if( 0 ) CERR << "create(*) implclass=["<<key<<"] allocated @" <<std::hex<<&v <<"\n";
			}
			catch(...)
			{
				return false;
			}
                        return 0 != &v;
                }
        }; // abstract_creator<T *>


} } // namespaces

#endif // s11n_ABSTRACT_CREATOR_H_INCLUDED

