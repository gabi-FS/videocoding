import os
import subprocess


path = '/home/gabriela/Documents/videos_teste' # change path

videos = os.listdir(r'/home/gabriela/Documents/videos_teste') # change path

for video in videos:
	#reduz o FPS dos vídeos
	video15 = video.replace('.y4m', '_15fps.y4m')
	fps15 = (fr'ffmpeg -i {os.path.join(path, video)} -filter:v fps=fps=15 {os.path.join(path, video15)}')
	os.system(fps15) 

	#interpola de volta para 60 fps só com MCI
	videoMCI = video15.replace('_15fps.y4m', '_recMCI.y4m')
	minterpolateMCI = (fr'ffmpeg -i {os.path.join(path, video15)} -filter:v "minterpolate=fps=30:mi_mode=mci:mb_size=16" {os.path.join(path, videoMCI)}')
	os.system(minterpolateMCI) 

	#interpola de volta para 60 fps com VSBMC
	videoVSBMC = video15.replace('_15fps.y4m', '_recVSBMC.y4m')
	minterpolateVSBMC = (fr'ffmpeg -i {os.path.join(path, video15)} -filter:v "minterpolate=fps=30:mi_mode=mci:vsbmc=1:mb_size=16" {os.path.join(path, videoVSBMC)}')
	os.system(minterpolateVSBMC) 
	
	print (fps15)
	print (minterpolateVSBMC)
	print (minterpolateMCI)


# create file 
'''
videos2 = os.listdir(r'/home/eclvc/gabifurtado/raw_videos') # change path


videos2.sort()
arquivo = open('resultados.txt', 'a')
for i in range(0, len(videos2), 4):
	original = videos2[i] 	
	novos = videos2[i+2:i+4]
	arquivo.write(f'*** {original} ***\n')
	for x in novos:
		#arquivo.write(f'- {x}\n')
		#psnr_ssim = (fr'ffmpeg -i {os.path.join(path, x)} -i {os.path.join(path, original)} -lavfi "[0][1]ssim;[0][1]psnr" -f null - |& tee >(grep Parsed_ >> {arquivo.name})')
		#os.system(psnr_ssim)
		#arquivo.write('\n')
		conteudos = []
		conteudos.append(f'- {x}\n')
		psnr_ssim = (fr'ffmpeg -i {os.path.join(path, x)} -i {os.path.join(path, original)} -lavfi "[0][1]ssim;[0][1]psnr" -f null -')
		output = ''
		output = subprocess.getoutput(psnr_ssim)
		indice = output.find('SSIM Y')
		conteudos.append(f'{output[indice:]}\n')
		conteudos.append('\n')
		arquivo.writelines(conteudos)        

	arquivo.write('--------------------------------------------------------------------------------------\n')

arquivo.close()
'''