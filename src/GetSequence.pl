use strict;
use File::Path qw(make_path);
use lib qw(BioPerl-1.7.4);
$| = 1;
use Getopt::Long;
use File::Spec;
use File::Basename;
use Bio::DB::Fasta;

my $seqs;

my $inputfile = $ARGV[0];
my $outputfile = $ARGV[1];
my $GENOMEFILE = $ARGV[2];

# 	$GENOMEFILE = "./data_resource/hg19/index";

my $seqDb =  Bio::DB::Fasta->new($GENOMEFILE);
print "seqDb info : $seqDb\n";
my @ids      = $seqDb->get_all_primary_ids;
print "id info : @ids\n";

my $name = (split /\/|\.bed/,$inputfile)[-1];
print "name : $name\n";
open (f1,"$inputfile") || die "Error $inputfile";
open (o1,">$outputfile") || die "Error";
while (<f1>)
{
	$_=~s/\s+$//;
	my @a = split /\t/,$_;
	my ($chr,$start,$end,@rest)=split /\t/,$_;
	next if $chr eq "chrY";
	print o1 ">$a[0]\-$a[1]\-$a[2]\-$a[4]\n";
	my $Seq = uc($seqDb->get_Seq_by_id($chr)->subseq($start=>$end));
	print o1 "$Seq\n";
}
close f1;
close o1;
print("finish GetSequence.pl\n");

