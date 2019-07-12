import os

os.environ['CANT_SINTESIS'] = '100' #si quiero dos cantidades distintas para cada uno, editar esta variable entre importaciones 
os.environ['SONO_BASE_DIR'] = os.path.join('sintetizados', 'sonogramas')
os.environ['AUDIO_BASE_DIR'] = os.path.join('sintetizados', 'audios')

print('Importando 1!')
import sint_benteveo_ruido
print('Importando 2!')
import sint_chingolo_ruido