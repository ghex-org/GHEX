#!/bin/bash
#
# Apply the formating to to the code base.

# Path to the file containing the formating
FORMAT_SPEC="$(git rev-parse --show-toplevel)/.clang-format"

# Sanatiy check.
if [ ! -e "${FORMAT_SPEC}" ]
then
	echo "The '.clang-format' specification does not exist." >&2
	exit 1
elif ! hash clang-format
then
	echo "'clang-format' is not installed." >&2
	exit 2
fi


print_help()
{
	cat << __EOH__
This script applies the formating specified in the '.clang-format' file to the code.
There are different ways of operations.
	$0 [ --all | --git-filter | files_to_format_inplace... ]

If called without arguments, it is equvalent to passing '--git-filter'.
The other arguments have the following meanings:
	--all 	Apply formating on the entier codebase.
	--git-filter 	Operate as git filter, the same as operating without
		arguments.

To use this script as git filter, you must run the following command:
	git config --local include.path "$(git rev-parse --show-toplevel)/.git_formater_config"
__EOH__
	exit $?
}


# Mode in which we operate.
#	0:	Act as `git` filter, the default, i.e. operate on the input stream.
#	1:	Act on the whole code base, i.e. run `find`.
#	2:	Act on a list of files passed by command line.
OPERATION_MODE=0

if [ "$#" -eq 0 ]
then
	# Nothing specified, use the default specification.
	:

elif [ "$#" -eq 1 ]
then
	case "$1" in
	  --all)
		OPERATION_MODE=1
	  ;;

	  --help|-h)
	  	print_help
	  	exit $?
	  ;;

	  --git-filter)
	  	# explicitly specified filtering mode.
	  	OPERATION_MODE=0
	  ;;

  	  -*)
  	  	echo "Unknown command '$1'" >&2
  	  	echo " Try using '--help'" >&2
  	  	exit 3
  	  ;;

	  *)
	  	# This might be the case that exactly one file is specified.
	  	#  So we check if the file exists.
	  	if [ ! -e "$1" ]
		then
			echo "Passed path to non existing file '$1'" >&2
			exit 4
		fi
		OPERATION_MODE=2
	  ;;
	esac

else
	# We have to process the set of files. We now make sure that they exist.
	for FILE_TO_CHECK in "$@"
	do
		if [ ! -e "${FILE_TO_CHECK}" ]
		then
			echo "Passed path to non existing file '$1'" >&2
			exit 4
		fi
	done
	OPERATION_MODE=2
fi


if [ "${OPERATION_MODE}" -eq 0 ]
then
	# Filtering is just reading from stdin and writing to stdout.
	exec clang-format "--style=file:${FORMAT_SPEC}"

elif [ "${OPERATION_MODE}" -eq 1 ]
then
	# Operate on all files.
	#  We only use a certain subset of them.
	find "$(git rev-parse --show-toplevel)/include" "$(git rev-parse --show-toplevel)/src" \( -name '*.h*' -o -name '*.c*' -o -name '*.cu' \) -exec clang-format "--style=file:${FORMAT_SPEC}" -i {} \;
	exit $?

else
	for FILE_TO_FORMAT in "$@"
	do
		clang-format "--style=file:${FORMAT_SPEC}" -i "${FILE_TO_FORMAT}"
		ERROR_CODE="$?"
		if [ "${ERROR_CODE}" -ne 0 ]
		then
			echo "Failed to format '${FILE_TO_FORMAT}'" >&2
			exit "${ERROR_CODE}"
		fi
	done
	exit 0
fi
