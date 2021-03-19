build:
	pip3 install --user -r requirements.txt
	mkdir -p morpho.nosync
	cd morpho.nosync && curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1836{/czech-morfflex-pdt-161115.zip} && unzip czech-morfflex-pdt-161115.zip
	cd morpho.nosync && curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0023-68D9-0{/english-morphium-wsj-140407.zip} && unzip english-morphium-wsj-140407.zip
	
clean:
	rm -rf morpho.nosync
	rm -rf model.nosync
	rm -rf data.nosync
	# this does NOT uninstall packages installed by pip3

