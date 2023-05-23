#!/bin/bash

if [ ! -f $PWD/LICENSE_HEADER ] ; then
    echo "you have to run the script ./tools/check_license.sh from the root of the sources"
    exit -1
fi

LICENSELEN=`wc -l LICENSE_HEADER | cut -f1 -d ' '`

for file in `git ls-tree -r $(git branch | grep \* | cut -d ' ' -f2) --name-only | grep -E ".*\.(h|hpp|cpp|cu|c|cpp.in)$"`; do
    if ! grep -q "Copyright (c) " $file ; then
        if [ "$1" == "fix" ]; then
            echo "$(cat LICENSE_HEADER)
$(cat ${file})" > ${file}
        else
            echo "no copyright in ${file} -- use fix"
        fi
    else
        if [ "$1" == "update" ]; then
            #head -$LICENSELEN ${file} | diff -u LICENSE_HEADER - | patch -R ${file}
            sed -i '0,/\*\//d' ${file}
            echo "$(cat LICENSE_HEADER)
$(cat ${file})" > ${file}
        else
            if [[ $(head -$LICENSELEN ${file} | diff -u LICENSE_HEADER -) ]]; then
                echo "found wrong copyright in ${file} -- use update"
            fi
        fi
    fi
done
