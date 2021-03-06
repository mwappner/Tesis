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

for ifile to numberOfFiles
	
	selectObject: files
	fileName$ = Get string: ifile
	specFile$ = replace$(fileName$, formato_de_audio$, formato_de_imagen$, 1)
	archivo = Read from file: read_directory$ + "/" + fileName$

	Select outer viewport: 0, 12, 0, 3.5
	View & Edit
	editor: archivo
		Spectrogram settings: 0, 10000, 0.008, 75
		Advanced spectrogram settings: 2000, 500, "Fourier", "Gaussian", "yes", 100, 6, 0
		Paint visible spectrogram: "yes", "no", "no", "no", "yes"
	endeditor
	removeObject: archivo

	Save as 600-dpi PNG file: write_directory$ + "/" + specFile$
	Erase all

endfor

selectObject: files
Remove
appendInfoLine: "Finalizado."