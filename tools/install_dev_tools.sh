#!/bin/bash

# from https://stackoverflow.com/questions/3915040/bash-fish-command-to-print-absolute-path-to-a-file/23002317#23002317
function abspath() {
    # generate absolute path from relative path
    # $1     : relative filename
    # return : absolute path
    if [ -d "$1" ]; then
        # dir
        (cd "$1"; pwd)
    elif [ -f "$1" ]; then
        # file
        if [[ $1 = /* ]]; then
            echo "$1"
        elif [[ $1 == */* ]]; then
            echo "$(cd "${1%/*}"; pwd)/${1##*/}"
        else
            echo "$(pwd)/$1"
        fi
    fi
}

thisfile=$(abspath ${0})
thisdir=$(dirname "${thisfile}")
currentdir0=$(pwd) 
currentdir=$(abspath ${currentdir0}) 

# clone and build uncrustify
mkdir -p ${thisdir}/uncrustify_tmp
cd ${thisdir}/uncrustify_tmp
git clone git@github.com:uncrustify/uncrustify.git
cd uncrustify
git checkout uncrustify-0.70.1
mkdir build
cd build
mkdir install
cmake \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    ..
make -j install
cp install/bin/uncrustify ${thisdir}
cd ${currentdir}
rm -rf ${thisdir}/uncrustify_tmp

# add git-hook
cd ${thisdir}
cd ../.git/hooks
githook=pre-commit
#ls hooks
echo "#!/bin/bash" > ${githook}
echo "python \\" >> ${githook}
echo "    \"${thisdir}/git_hooks/git-uncrustify.py\" \\" >> ${githook}
echo "    \"--commit\" \\" >> ${githook}
echo "    \"-c=${thisdir}/uncrustify.cfg\" \\" >> ${githook}
echo "    \"-bin=${thisdir}/uncrustify\"" >> ${githook}
cd ${currentdir}


