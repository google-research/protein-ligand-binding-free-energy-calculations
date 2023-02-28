#!/bin/bash



for lig in $( ls -d ligand_*/ ); do
	cd $lig
	mkdir ions
	cp ../apo_protein/ions/*itp ions/
	grep "MG\|ZN" protein/protein.pdb > ions/ions.pdb
    grep -v "MG\|ZN" protein/protein.pdb > protein/_protein.pdb
    mv protein/_protein.pdb protein/protein.pdb
    cd ../
 done