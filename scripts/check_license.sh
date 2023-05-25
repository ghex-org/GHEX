#!/bin/bash

if [ ! -f $PWD/LICENSE_HEADER ] ; then
    echo "you have to run the script ./tools/check_license.sh from the root of the sources"
    exit -1
fi

LICENSELEN=`wc -l LICENSE_HEADER | cut -f1 -d ' '`

# c++ source
for file in `git ls-tree -r $(git branch | grep \* | cut -d ' ' -f2) --name-only | grep -E ".*\.(h|hpp|cpp|cu|c|cpp.in|hpp.in)$"`; do
    if [[ $(head -$LICENSELEN ${file} | diff -u LICENSE_HEADER -) ]]; then
        if [ "$1" == "fix" ]; then
            if ! grep -q "Copyright (c) " $file ; then
                echo "$(cat LICENSE_HEADER)
$(cat ${file})" > ${file}
            else
                sed -i '0,/\*\//d' ${file}
                echo "$(cat LICENSE_HEADER)
$(cat ${file})" > ${file}
            fi
        else
            echo "found wrong copyright in ${file} -- use fix"
        fi
    fi
done

# fortran source
LICENSE_HEADER_FORTRAN=$(sed 's/.\*\/*/!/g' LICENSE_HEADER)
for file in `git ls-tree -r $(git branch | grep \* | cut -d ' ' -f2) --name-only | grep -E ".*\.(f90|mod|mod.in|f90.in)$"`; do
    if [[ $(head -$LICENSELEN ${file} | diff -u - <(echo "${LICENSE_HEADER_FORTRAN}")) ]]; then
        if [ "$1" == "fix" ]; then
            if ! grep -q "Copyright (c) " $file ; then
                echo "${LICENSE_HEADER_FORTRAN}
$(cat ${file})" > ${file}
            else
                tmpfile=$(mktemp)
                cp ${file} "$tmpfile" && awk -v ctr=0 '{if(ctr>0 || (!/^ *!/)){ctr=ctr+1 ; print}}' "$tmpfile" > ${file}
                echo "${LICENSE_HEADER_FORTRAN}
$(cat ${file})" > ${file}
            fi
        else
            echo "found wrong copyright in ${file} -- use fix"
        fi
    fi
done

# python source
LICENSE_HEADER_PYTHON=$(sed 's/.\*\/*/#/g' LICENSE_HEADER)
for file in `git ls-tree -r $(git branch | grep \* | cut -d ' ' -f2) --name-only | grep -E ".*\.(py|py.in)$"`; do
    if [[ $(head -$LICENSELEN ${file} | diff -u - <(echo "${LICENSE_HEADER_PYTHON}")) ]]; then
        if [ "$1" == "fix" ]; then
            if ! grep -q "Copyright (c) " $file ; then
                echo "${LICENSE_HEADER_PYTHON}
$(cat ${file})" > ${file}
            else
                tmpfile=$(mktemp)
                cp ${file} "$tmpfile" && awk -v ctr=0 '{if(ctr>0 || (!/^ *#/)){ctr=ctr+1 ; print}}' "$tmpfile" > ${file}
                echo "${LICENSE_HEADER_PYTHON}
$(cat ${file})" > ${file}
            fi
        else
            echo "found wrong copyright in ${file} -- use fix"
        fi
    fi
done
