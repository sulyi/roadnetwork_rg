#!/bin/sh
set -euo pipefail

# TODO: add options to set options in runtime

VENV=vgh-pagesenv

BRANCH_NAME=gh-pages
DOCSRC=src/doc
DOCBUILD=build/html

usage()
{
  echo "Usage: $0"
  cat <<HELPME
Creates a new git branch containing documentation only generated by Sphinx.

Requires sphinx to be installed in virtual environment set by VENV.
Builds documentation source set by DOCSRC.
Expects result to be in location set by DOCBUILD.

All paths relative to current directory (not to script location).

Name of branch created is set by BRANCH_NAME.

Options:
    --help    Prints this message.
    --auto    Runs without any user input, (might want to make a backup first and make sure 'origin' is set)

WIP

As of now variables set inside script, argument options are coming...
HELPME
}

require_clean_work_tree()
{
  # according https://stackoverflow.com/a/3879077/3417742
  # Update the index
  git update-index -q --ignore-submodules --refresh
  err=0

  # Disallow unstaged changes in the working tree
  if ! git diff-files --quiet --ignore-submodules --
  then
    echo >&2 "Cannot $1. Reason: you have unstaged changes:"
    git diff-files --name-status -r --ignore-submodules -- >&2
    err=1
  fi

  # Disallow uncommitted changes in the index
  if ! git diff-index --cached --quiet HEAD --ignore-submodules --
  then
    echo >&2 "Cannot $1. Reason: your index contains uncommitted changes:"
    git diff-index --cached --name-status -r --ignore-submodules HEAD -- >&2
    err=1
  fi

  if [ $err = 1 ]
  then
    echo >&2 "\nPlease commit or stash them.\n"
    exit 1
  fi
}

# MAIN

# do setup (read arguments)
AUTO=0
POSITIONAL=()

while [[ $# -gt 0 ]]
do
  key="$1"

  case $key in
  -h|--help)
    usage
    exit 0
  ;;
  --auto)
    AUTO=1
    shift
  ;;
  *)
    if [[ $1 = -* ]]
    then
      echo 'Unexpected option found'
      usage
      exit 1
    else
      # XXX: see usage WIP section
      POSITIONAL+=("$1")
      shift
      echo 'Unexpected argument found'
      usage
      exit 1
    fi
  ;;
  esac
done

[ $AUTO -eq 0 ] && require_clean_work_tree "create $BRANCH_NAME"

virtualenv $VENV
source $VENV/bin/activate
pip install -r requirements.txt

GITVERSION=`git describe --long --dirty=-dev0`
PACKAGEVERSION=`python setup.py --version`

echo "$PACKAGEVERSION is tagged $GITVERSION"

python setup.py install

# build doc
cd $DOCSRC
make clean && make html
cd -

# create/update branch
if [ $(git branch --list "$BRANCH_NAME") ]
then
  git checkout $BRANCH_NAME
  if [ $AUTO -eq 0 ]
  then
    # check if origin exists
    git config --get remote.origin.url >/dev/null &&
      git pull  --ff-only origin $BRANCH_NAME ||
      echo -e '\nMissing remote 'origin' ... skipping pull\n'
  else
    git pull origin $BRANCH_NAME
  fi
else
  git checkout --orphan $BRANCH_NAME
fi

if [ -d "$DOCBUILD" ]
then
  git reset --hard
  git ls-files -z --others --exclude-standard --exclude=/$DOCBUILD/ |
    xargs -0 rm -rf
  cp -r $DOCBUILD/. . && rm -rf $DOCBUILD
  git add .
  git clean -fd  # remove empty directories, not that they matter
  git commit -am "Rebuild from $GITVERSION"
else
  echo "Failed to locate build directory"
fi

# clean up
# TODO: move to trap
rm -r $VENV

echo -e "\n\nBuilding *$BRANCH_NAME* was successful," \
        "\nplease consider squashing before pushing!\n"
