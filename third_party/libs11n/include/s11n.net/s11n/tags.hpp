#ifndef s11n_net_s11n_1_1_TAGS_HPP_INCLUDED
#define s11n_net_s11n_1_1_TAGS_HPP_INCLUDED 1
// Defines some "tag" structs for some functional composition experimentation.
namespace s11n {



	/**
	   A tag type for serialization functors.
	*/
	struct serialize_f_tag {};

	/**
	   A tag type for serialization functors which look like:

	   bool operator()() const
	*/
	struct serialize_nullary_f_tag : serialize_f_tag {};

	/**
	   A tag type for serialization functors which look like:

	   bool operator()( NodeT &, SerializableT const & ) const
	*/
	struct serialize_binary_f_tag : serialize_f_tag {};

	/**
	   A tag type for serialization functors which look like:

	   bool operator()( NodeT & ) const
	*/
	struct serialize_unary_node_f_tag : serialize_f_tag {};

	/**
	   A tag type for serialization functors which look like:

	   bool operator()( SerializableT const & ) const
	*/
	struct serialize_unary_serializable_f_tag : serialize_f_tag {};


	/**
	   A tag type for deserialization functors.
	*/
	struct deserialize_f_tag {};

	/**
	   A tag type for deserialization functors which look like:

	   bool operator()() const
	*/
	struct deserialize_nullary_f_tag : deserialize_f_tag {};

	/**
	   A tag type for deserialization functors which look like:

	   bool operator()( NodeT const &, SerializableT & ) const
	*/
	struct deserialize_binary_f_tag : deserialize_f_tag {};

	/**
	   A tag type for deserialization functors which look like:

	   SerializableT * operator()( NodeT const & ) const
	*/
	struct deserialize_unary_node_f_tag : deserialize_f_tag {};

	/**
	   A tag type for deserialization functors which look like:

	   bool operator()( SerializableT & ) const
	*/
	struct deserialize_unary_serializable_f_tag : deserialize_f_tag {};


} // namespace s11n


#endif // s11n_net_s11n_1_1_TAGS_HPP_INCLUDED

