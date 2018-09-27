writeInfoLine: "Iniciando..."
directory$ = "/home/marcos/Documents/Audios/Finches"
strings = Create Strings as file list: "list", directory$ + "/*.wav"
numberOfFiles = Get number of strings
f_viejo$ = ".wav"
f_nuevo$ = ".png"
for ifile to numberOfFiles
    selectObject: strings
    fileName$ = Get string: ifile
    nuevo$ = replace$(fileName$, f_viejo$, f_nuevo$, 1)
    appendInfoLine: directory$ + "/" + nuevo$
endfor
 selectObject: strings
Remove