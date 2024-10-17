#ifndef cliutil_ARGV_PARSER_HPP
#define cliutil_ARGV_PARSER_HPP 1

namespace cliutil {

	typedef std::map<std::string,std::string> argv_map;
	typedef std::vector<std::string> unnamed_args_list;


	/**

	Scans argv for arguments in these forms:

	a) -KEY VALUE

	b) -KEY=VALUE

	c) -KEY

	d) UNNAMED_ARGUMENT (e.g., a filename)

	Form (c) represents boolean flags, and such a flag is
	considered to have a true value.

	The special argument '--' means "stop scanning".

	Forms (a), (b), and (c) add their arguments to the target map,
	minus any leading '-' characters. Form (d) adds an entry as-is
	to target2.

	Returns the number of named arguments parsed, or -1 if
	startAtPos is greater than or equal to argc.

	*/
        int
        parse_args( int argc, char ** argv, int startAtPos, argv_map & target, unnamed_args_list & target2 );

} // namespace

#endif // cliutil_ARGV_PARSER_HPP
