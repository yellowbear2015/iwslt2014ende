#!/bin/sh


if [ $# -ne 3 ];then
	echo "Usage: sh tokenize_udf.sh [en|zh|ru|...] [input.txt] [output.txt] " >&2
	exit
fi

lang=$1
infile=$2
outfile=$3

islower=true
ne=true


idx=`shuf -i 1-100000|head -n1`_`date '+%s'`

in_tb=icbu_translate_dev.tmp_sentence_for_tok_${lang}_${idx}
out_tb=icbu_translate_dev.tmp_sentence_for_tok_${lang}_${idx}_tokened

cat $infile |awk -F '\005' '{s=$1;for(i=2;i<=NF;++i) s=s" "$i; print NR"\005"s}' > ${infile}.tmp.$idx

odpscmd -e " drop table if exists $in_tb purge;
	create table $in_tb (idx bigint, text string)lifecycle 2;
	tunnel upload -fd='\u0005' ${infile}.tmp.$idx $in_tb;

	set odps.sql.mapper.split.size=8;
	drop table if exists $out_tb purge;
	create table $out_tb lifecycle 2
	as
	select idx, udf_nlptk('tokenize',text,'$lang','*',$islower,$ne,false) as tok
	from $in_tb;

	tunnel download -fd='\u0009' $out_tb ${infile}.tmp.tokend.$idx ;
"

odpscmd -e "
	drop table if exists $in_tb purge;
	drop table if exists $out_tb purge;
" &

cat ${infile}.tmp.tokend.$idx |sort --parallel=12 --buffer-size=3000M -T . -k 1 -n |awk -F '\011' '{print $2}' > $outfile

rm -rf ${infile}.tmp.$idx &
rm -rf ${infile}.tmp.tokend.$idx &
wait

