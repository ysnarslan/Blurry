import os
import shutil
import time

# Temizlenecek klasörün yolu
folder_path = r'static/images'


while True:

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Error: {e}')
    print('Klasör basariyla temizlendi.')
        
    # 24 saat beklenir ve yeniden kontrol edilir
    time.sleep(86400)