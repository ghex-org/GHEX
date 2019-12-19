#!/usr/bin/python

# adapted from
# https://github.com/BlueBrain/git-cmake-format

from __future__ import print_function
import os
import subprocess
import sys
import time

Git='git'
Diff='diff'
Sed='sed'
ClangFormat='clang-format'
Style='-style=file'
FormatAll=False
IgnoreList=[]
ExtensionList=['.h', '.cpp', '.hpp', '.c', '.cc', '.hh', '.cxx', '.hxx', '.cu', '.m']

def getGitHead():
    RevParse = subprocess.Popen([Git, 'rev-parse', '--verify', 'HEAD'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    RevParse.communicate()
    if RevParse.returncode:
        return '4b825dc642cb6eb9a060e54bf8d69288fbee4904'
    else:
        return 'HEAD'

def getGitRoot():
    RevParse = subprocess.Popen([Git, 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return RevParse.stdout.read().strip()

def getAllFiles():
    GitLs = subprocess.Popen([Git, "ls-files"], stdout=subprocess.PIPE)
    GitLsRet = GitLs.stdout.read().strip()
    GitLsRet = GitLsRet.decode()
    return GitLsRet.split('\n')

def getEditedFiles(InPlace):
    Head = getGitHead()
    GitArgs = [Git, 'diff-index']
    if not InPlace:
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
    proc = subprocess.Popen([ClangFormat, Style, '-i', os.path.join(GitRoot,FileName)])
    # wait for the operations to complete
    proc.communicate()
    return

def patchFile(FileName,PatchFile):
    # print changed file
    GitShowRet = subprocess.Popen([Git, "show", ":" + FileName], stdout=subprocess.PIPE)
    # pipe it to clang-format
    ClangFormatRet = subprocess.Popen([ClangFormat, Style], stdin=GitShowRet.stdout, stdout=subprocess.PIPE)
    # pipe output to diff and compare with original file
    DiffRet = subprocess.Popen([Diff, "-u", FileName, "-"], stdin=ClangFormatRet.stdout, stdout=subprocess.PIPE)
    # replace diff annotations to make it a git patch by piping the output to sed; redirect the results to the PatchFile
    SedRet = subprocess.Popen([Sed, "-e", "1s|--- |--- a/|", "-e", "2s|+++ -|+++ b/" + FileName + "|"], stdin=DiffRet.stdout, stdout=PatchFile)
    # wait for the operations to complete
    SedRet.communicate()
    return

def printUsageAndExit():
    print("Usage: " + sys.argv[0] + " [--pre-commit|--cmake] [--all]" +
          "[<path/to/git>] [<path/to/clang-format>]  [<path/to/diff>]  [<path/to/sed>]")
    sys.exit(1)

if __name__ == "__main__":
    if 2 > len(sys.argv):
        printUsageAndExit()

    if "--pre-commit" == sys.argv[1]:
        InPlace = False
    elif "--cmake" == sys.argv[1]:
        InPlace = True
    else:
        printUsageAndExit()

    for arg in sys.argv[2:]:
        if "git" in arg:
            Git = arg
        elif "clang-format" in arg:
            ClangFormat = arg
        elif "diff" in arg:
            Diff = arg
        elif "sed" in arg:
            Sed = arg
        elif "-style=" in arg:
            Style = arg
        elif "-ignore=" in arg:
            IgnoreList = arg.strip("-ignore=").split(";")
        elif "--all" in arg:
            FormatAll=True
        else:
            printUsageAndExit()

    # get a list of changed files which were changed

    if (FormatAll):
        EditedFiles = getAllFiles()
    else:
        EditedFiles = getEditedFiles(InPlace)

    ReturnCode = 0

    # clang-format in-place (not used in the git-hook)
    if InPlace:
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
    for FileName in EditedFiles:
        if not isFormattable(FileName):
            continue
        patchFile(FileName,f)

    # check whether the git patch is empty
    f.seek(0)
    if not f.read(1):
        f.close()
        print("Files in this commit comply with the clang-format rules.")
        os.remove(PatchName)
    else:
        f.seek(0)
        ReturnCode = 1
        print("The following differences wre found between the code to commit")
        print("and the clang-format rules:")
        print()
        print(f.read())
        f.close()
        print("Files in this commit do not compy with the clang-format rules.")
        print("You can apply these changes with:")
        print("  git apply --index ", PatchName)
        print("(may need to be called from the root directory of your repository)")
        print("or you can run")
        print("  git clang-format")
        print("if you want to apply clang-format to unstaged files as well, run")
        print("  git clang-format -f")
        print("note that git clang-format will only affect the changed lines")

    sys.exit(ReturnCode)

