form Archivo de lectura y de escritura
	sentence Directorio_de_entrada
	sentence Directorio_de_salida
	sentence Archivo
endform

archivo_de_entrada$ = directorio_de_entrada$ + "\" + archivo$ + ".mp3"
archivo_de_salida$ = directorio_de_salida$ + "\" + archivo$ + ".png"

archivo = Read from file: archivo_de_entrada$
spec = To Spectrogram: 0.005, 5000, 0.002, 20, "Gaussian"
selectObject: spec
Select outer viewport: 0, 12, 0, 3.5
Paint: 0, 0, 0, 0, 100, "yes", 50, 6, 0, "yes"
Save as 600-dpi PNG file: archivo_de_salida$
Erase all
selectObject: archivo
plusObject: spec
Remove


