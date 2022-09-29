import os
import subprocess

path = 'teste_dir'
#psnr_ssim = (fr'ffmpeg -i teste_dir/bowing_cif_recMCI.y4m -i teste_dir/bowing_cif.y4m -lavfi "[0][1]ssim;[0][1]psnr" -f null - 2>&1 | grep Parsed_ >> escrita.txt')
#os.system(psnr_ssim)
'''
videos2 = os.listdir(r'/home/gabrielafs/Documentos/UFSC/Video_coding/teste_dir')
videos2.sort()
arquivo = open('escrita.txt', 'a')
for i in range(0, len(os.popen('cat /etc/services').read()videos2), 4):
	original = videos2[i] 
	novos = videos2[i+2:i+4]
	arquivo.write(f'*** {original} ***\n')
	for x in novos:
		arquivo.write(f'- {x}\n')
		psnr_ssim = (fr'ffmpeg -i {os.path.join(path, x)} -i {os.path.join(path, original)} -lavfi "[0][1]ssim;[0][1]psnr" -f null - 2>&1 | grep Parsed_ >> {arquivo.name}')
		os.system(psnr_ssim)
		arquivo.write('\n')

	arquivo.write('--------------------------------------------------------------------------------------')

arquivo.close()'''

videos2 = os.listdir(r'/home/eclvc/gabifurtado/teste_dir')
videos2.sort()
nome = 'escrita.txt'
arquivo = open(nome, 'a')
for i in range(0, len(videos2), 4):
	original = videos2[i]
	novos = videos2[i+2:i+4]
	arquivo.write(f'*** {original} *** \n ')
	for x in novos:
		conteudos = []
		conteudos.append(f'- {x}\n')
		psnr_ssim = (fr'ffmpeg -i {os.path.join(path, x)} -i {os.path.join(path, original)} -lavfi "[0][1]ssim;[0][1]psnr" -f null -')
		output = ''
		output = subprocess.getoutput(psnr_ssim)
		indice = output.find('SSIM Y')
		conteudos.append(f'{output[indice:]}\n')
		conteudos.append('\n')
		arquivo.writelines(conteudos)
        #output = subprocess.Popen(['Parsed_'], bufsize=0, stdout=arquivo)
		
        #indice = output.find('Parsed_')
        

	arquivo = open(nome, 'a')
	arquivo.write('--------------------------------------------------------------------------------------\n')

arquivo.close()