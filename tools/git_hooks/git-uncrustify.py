#!/usr/bin/python

# adapted from
# https://github.com/BlueBrain/git-cmake-format

from __future__ import print_function
import os
import subprocess
import sys
import time

#Uncrustify='uncrustify'
Uncrustify='/home/boeschf/Development/GHEX_fork/tools/uncrustify'
Config=''
Git='git'
Diff='diff'
Sed='sed'
FormatAll=False
IgnoreList=[]
ExtensionList=['.h', '.cpp', '.hpp', '.c', '.cc', '.hh', '.cxx', '.hxx', '.cu', '.m']

def getGitRoot():
    RevParse = subprocess.Popen([Git, 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return RevParse.stdout.read().strip()

def getGitDir():
    RevParse = subprocess.Popen([Git, 'rev-parse', '--git-dir'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return RevParse.stdout.read().strip()

def getGitHead():
    RevParse = subprocess.Popen([Git, 'rev-parse', '--verify', 'HEAD'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    RevParse.communicate()
    if RevParse.returncode:
        return '4b825dc642cb6eb9a060e54bf8d69288fbee4904'
    else:
        return 'HEAD'

def getAllFiles():
    GitDir=getGitDir()
    GitLs = subprocess.Popen([Git, "--git-dir", GitDir, "ls-files"], stdout=subprocess.PIPE)
    GitLsRet = GitLs.stdout.read().strip()
    GitLsRet = GitLsRet.decode()
    return GitLsRet.split('\n')

def getEditedFiles():
    Head = getGitHead()
    GitArgs = [Git, 'diff-index']
    GitArgs.append('--cached')
    GitArgs.extend(['--diff-filter=ACMR', '--name-only', Head])
    DiffIndex = subprocess.Popen(GitArgs, stdout=subprocess.PIPE)
    DiffIndexRet = DiffIndex.stdout.read().strip()
    DiffIndexRet = DiffIndexRet.decode()
    return DiffIndexRet.split('\n')

def isFormattable(File):
    for Dir in IgnoreList:
        if '' != Dir and '' != os.path.commonprefix([os.path.relpath(File), os.path.relpath(Dir)]):
            return False
    Extension = os.path.splitext(File)[1]
    for Ext in ExtensionList:
        if Ext == Extension:
            return True
    return False

def formatFile(FileName, GitRoot):
    # change the FileName in place
    proc = subprocess.Popen([Uncrustify, "-c", Config, "-l cpp", "--replace", "--no-backup", "-q", os.path.join(GitRoot,FileName)])
    # wait for the operations to complete
    proc.communicate()
    return

def patchFile(FileName, GitRoot, PatchFile):
    # print changed file
    GitShowRet = subprocess.Popen([Git, "show", ":" + FileName], stdout=subprocess.PIPE)
    # pipe it to uncrustify
    UncrustifyRet = subprocess.Popen([Uncrustify, "-q", "-l", "cpp", "-c", Config], stdin=GitShowRet.stdout, stdout=subprocess.PIPE)
    # pipe output to diff and compare with original file
    FileName2 = os.path.join(GitRoot,FileName)
    DiffRet = subprocess.Popen([Diff, "-u", FileName2, "-"], stdin=UncrustifyRet.stdout, stdout=subprocess.PIPE)
    # replace diff annotations to make it a git patch by piping the output to sed; redirect the results to the PatchFile
    SedRet = subprocess.Popen([Sed, "-e", "1s|--- |--- a/|", "-e", "2s|+++ -|+++ b/" + FileName + "|"], stdin=DiffRet.stdout, stdout=PatchFile)
    # wait for the operations to complete
    SedRet.communicate()
    return

def printUsageAndExit():
    print("Usage: " + sys.argv[0] + " --commit|--all" + " -c=<path/to/config>" +
          "\n  [-bin=<path/to/uncrustify>]" +
          "\n  [-git=<path/to/git>]" +
          "\n  [-diff=<path/to/diff>]" +
          "\n  [-sed=<path/to/sed>]" +
          "\n  [-ignore=<list;of;files>]")
    sys.exit(1)

if __name__ == "__main__":
    if 3 > len(sys.argv):
        printUsageAndExit()

    if "--commit" == sys.argv[1]:
        FormatAll = False
    elif "--all" == sys.argv[1]:
        FormatAll = True
    else:
        printUsageAndExit()
    
    if "-c" in sys.argv[2]:
        Config = sys.argv[2].strip("-c=")
        print(Config)
    else:
        printUsageAndExit()

    for arg in sys.argv[3:]:
        if "-bin=" in arg:
            Uncrustify = arg.strip("-bin=")
            print(Uncrustify)
        elif "-git=" in arg:
            Git = arg.strip("-git=")
        elif "-diff=" in arg:
            Diff = arg.strip("-diff=")
        elif "-sed=" in arg:
            Sed = arg.strip("-sed=")
        elif "-ignore=" in arg:
            IgnoreList = arg.strip("-ignore=").split(";")
        else:
            printUsageAndExit()

    # get a list of changed files which were changed
    if FormatAll:
        EditedFiles = getAllFiles()
    else:
        EditedFiles = getEditedFiles()

    print(EditedFiles)

    ReturnCode = 0

    if FormatAll:
        GitRoot = getGitRoot()
        for FileName in EditedFiles:
            if isFormattable(FileName):
                formatFile(FileName,GitRoot)
        sys.exit(ReturnCode)

    # create a temporary file for the git patch
    Prefix = "pre-commit-clang-format"
    Suffix = time.strftime("%Y%m%d-%H%M%S")
    PatchName = "/tmp/" + Prefix + "-" + Suffix + ".patch"
    f = open(PatchName, "w+")

    # add patch instructions for each file to the git patch
    GitRoot = getGitRoot()
    for FileName in EditedFiles:
        if not isFormattable(FileName):
            continue
        patchFile(FileName,GitRoot,f)

    # check whether the git patch is empty
    f.seek(0)
    if not f.read(1):
        f.close()
        print("Files in this commit comply with the format rules.")
        os.remove(PatchName)
    else:
        f.seek(0)
        ReturnCode = 1
        print("The following differences wre found between the code to commit")
        print("and the format rules:")
        print()
        print(f.read())
        f.close()
        print("Files in this commit do not compy with the format rules.")
        print("You can apply these changes with:")
        print("  git apply --index ", PatchName)
        print("(may need to be called from the root directory of your repository)")

    sys.exit(ReturnCode)
