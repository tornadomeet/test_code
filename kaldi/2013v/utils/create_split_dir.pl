#!/usr/bin/perl

# Copyright 2013  Guoguo Chen
# Apache 2.0.
#
# This script creates storage directories on different file systems, and creates
# symbolic links to those directories. For example, a command
#
#   utils/create_split_dir.pl /export/gpu-0{3,4,5}/egs/storage egs/storage
#
# will mkdir -p all of those directories, and will create links
#
#   egs/storage/1 -> /export/gpu-03/egs/storage
#   egs/storage/2 -> /export/gpu-03/egs/storage
#   ...
#
use strict;
use warnings;
use File::Spec;
use Getopt::Long;

my $Usage = <<EOU;
This script creates storage directories on different file systems, and creates
symbolic links to those directories.

Usage: utils/create_split_dir.pl <actual_storage_dirs> <pseudo_storage_dir>
 e.g.: utils/create_split_dir.pl /export/gpu-0{3,4,5}/egs/storage egs/storage

Allowed options:
  --suffix    : Common suffix to <actual_storage_dirs>    (string, default = "")

EOU

my $suffix="";
GetOptions('suffix=s' => \$suffix);

if (@ARGV < 2) {
  die $Usage;
}

my $ans = 1;

my $dir = pop(@ARGV);
system("mkdir -p $dir 2>/dev/null");
my $index = 1;
foreach my $file (@ARGV) {
  $file = $file . "/" . $suffix;
  my $actual_storage = File::Spec->rel2abs($file);
  my $pseudo_storage = "$dir/$index";

  # If the symbolic link already exists, delete it.
  if (-l $pseudo_storage) {
    unlink($pseudo_storage);
  }

  # Create the destination directory and make the link.
  system("mkdir -p $actual_storage 2>/dev/null");
  my $ret = symlink($actual_storage, $pseudo_storage);

  # Process the returned values
  $ans = $ans && $ret;
  if (! $ret) {
    print STDERR "Error linking $actual_storage to $pseudo_storage\n";
  }

  $index++;
}

exit($ans == 1 ? 0 : 1);
