#!/bin/sh

if [ "$1" = "SPL" ]
then
	if [ "$2" = "True" ]
	then
		xvfb-run -a java -cp "lib/*" au.edu.unimelb.services.ServiceProvider "SMD" 0.1 0.0 "false" $3 $4
	else
		xvfb-run -a java -cp "lib/*" au.edu.unimelb.services.ServiceProvider "SMD" 0.1 0.0 "false" $3 $4
	fi
else
	export USE_FILTER=$2
	export IMPORTLOG=$3
	export EXPORTMODEL=$4.pnml
	xvfb-run - a java -da -Xmx8G -classpath "lib/*" -Djava.library.path=./lib -Djava.util.Arrays.useLegacyMergeSort=true org.processmining.contexts.cli.CLI -f scripts/ILP.txt
fi