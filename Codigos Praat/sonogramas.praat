form Ubicación de archivos de lectura y destino de los sonogramas creados
	sentence Read_directory /home/marcos/Documents/Audios/Finches
	sentence Formato_de_audio  .wav
	sentence Write_directory /home/marcos/Documents/Codigos/Analisis cantos/Sonogramas
endform

writeInfoLine: "Iniciando."

files = Create Strings as file list: "list", read_directory$ + "/*" + formato_de_audio$
numberOfFiles = Get number of strings
appendInfoLine: "Analizando ", numberOfFiles, " archivos"

formato_de_imagen$ = ".png"
for ifile to 3
	selectObject: files
	fileName$ = Get string: ifile
	specFile$ = replace$(fileName$, formato_de_audio$, formato_de_imagen$, 1)
	archivo = Read from file: read_directory$ + "/" + fileName$

	spec = To Spectrogram: 0.005, 5000, 0.002, 20, "Gaussian"
	selectObject: spec
	Select outer viewport: 0, 12, 0, 3.5
	Paint: 0, 0, 0, 0, 100, "yes", 50, 6, 0, "yes"
	Save as 600-dpi PNG file: write_directory$ + "/" + specFile$
	Erase all
	removeObject: archivo, spec

endfor
selectObject: files
Remove