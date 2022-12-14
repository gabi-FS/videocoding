from copy import copy
from typing import Tuple
from matplotlib import test
import numpy as np
import matplotlib.pyplot as matp
from PIL import Image
# import scipy.ndimage
import skimage
import os


class VideoParser():
    def __init__(self, file_name, height=0, width=0, fps='', y_size=0, cr_size=0, cb_size=0) -> None:
        self.__file_name = file_name
        self.__height = height
        self.__width = width
        self.__fps = fps
        self.__end_header = 0
        self.__size = [y_size, cb_size, cr_size]
        if height == 0:
            self.__end_header = self.get_header_info()

    @property
    def file_name(self) -> str:
        return self.__file_name

    @property
    def height(self):
        return self.__height

    @property
    def width(self):
        return self.__width

    @property
    def fps(self) -> str:
        return self.__fps

    @property
    def size(self) -> list:
        return self.__size

    def get_header_info(self):
        try:
            with open(self.__file_name, 'rb') as f:
                a = f.readline()
                w = a[a.find(ord('W'))+1:]
                w = w[:w.find(ord(' '))]
                self.__width = int(w)
                h = a[a.find(ord('H'))+1:]
                h = h[:h.find(ord(' '))]
                self.__height = int(h)
                fps = a[a.find(ord('F'))+1:]
                fps = fps[:fps.find(ord(' '))]
                self.__fps = fps
                print(self.__height, self.__width, fps)
                end = f.tell()

                self.__size[0] = self.__height*self.__width
                f.seek(6+self.__size[0], 1)
                self.__size[1] = int((len(f.readline())-6)/2)
                self.__size[2] = self.__size[1]
                print(self.__size)
                return end

        except Exception as e:
            print(f'Erro: {e}')

    def get_frame(self, num_frame: int):
        '''input: file name, frame's number; returns a tuple of numpy.array video samples'''
        dtype = np.dtype('B')
        if num_frame < 1:
            print('Frame number must be greater than 0')
            return
        try:
            with open(self.__file_name, 'rb') as f:
                f.seek(self.__end_header)
                f.seek(6*num_frame + int((num_frame-1) *
                       self.__height*self.__width*1.5), 1)
                sample_y = np.frombuffer(
                    f.read(int(self.__height*self.__width)), dtype=dtype)
                sample_cb = np.frombuffer(
                    f.read(int(self.__height*self.__width/4)), dtype=dtype)
                sample_cr = np.frombuffer(
                    f.read(int(self.__height*self.__width/4)), dtype=dtype)
                return sample_y, sample_cb, sample_cr

        except Exception as e:
            print(f'Error: {e}')

    def plot_component_frame(self, num_frame, filename=' ', sample=0):
        '''Extracts the chosen frame from the raw file and creates a .pdf in grayscale using
        the given sample and filename. Sample: inform 0 for Y, 1 for Cb and 2 for Cr; if not informed, Y is used.'''
        if not (0 <= sample <= 2):
            print('Sample value must be between 0 and 2.')
            return
        component_sample = self.get_frame(num_frame)[sample]
        print(self.__size[0], self.__size[sample])
        print((self.__size[0]/self.__size[sample]))
        print(self.__height)
        if sample > 0:
            im = self.component_to_image(
                component_sample, (self.__size[0]/self.__size[sample])/2)
        else:
            im = self.component_to_image(component_sample)
        matp.figure()
        if filename == ' ':
            filename = self.__file_name[:len(
                self.__file_name)-4] + '_' + str(num_frame) + '_' + str(sample)
        matp.imshow(im, cmap="gray", aspect="auto")
        matp.savefig(f'{filename}.pdf')
        matp.close()

    def plot_frame(self, num_frame, filename=' '):
        '''Creates a .png from the chosen frame, using Image from PIL.'''
        try:
            frame = self.get_frame(num_frame)
            im = self.frame_to_image(frame)
            im = im.convert("RGB")
            if filename == ' ':
                filename = self.__file_name[:len(
                    self.__file_name)-4] + '_' + str(num_frame)
            im.save(f'{filename}.png')
        except Exception as e:
            print('Something went wrong!')
            print(e)

    def component_to_image(self, component_sample, sampling=1):
        component_sample = np.reshape(component_sample, (int(
            self.__height/sampling), int(self.__width/sampling)))
        im = Image.fromarray(np.uint8(component_sample), "L")
        return im

    def frame_to_image(self, frame):
        y, cb, cr = frame
        ups_cb, ups_cr = cb, cr
        if (self.__size[0]/self.__size[1]) == 4:
            y = np.reshape(y, (int(self.__height), int(self.__width)))
            cb = np.reshape(cb, (int(self.__height/2), int(self.__width/2)))
            cr = np.reshape(cr, (int(self.__height/2), int(self.__width/2)))
            ups_cb = np.kron(cb, np.ones((2, 2), dtype=np.dtype('B')))
            ups_cr = np.kron(cr, np.ones((2, 2), dtype=np.dtype('B')))
        try:
            frame = np.stack((y, ups_cb, ups_cr), axis=2)
            frame = np.reshape(frame, (self.__height, self.__width, 3))
            print(frame.shape)
            print(len(frame))
            im = Image.fromarray(np.uint8(frame), "YCbCr")
            return im
        except Exception as e:
            print('Something went wrong!')
            print(e)

    @staticmethod
    def get_psnr(original_array, test_array, component=0):
        '''Calculates PSNR metric for each frame in original and test arrays 
        and returns it. If component is -1, it will return PSNR for all channels; 
        otherwise, inform 0 for Y, 1 for Cb and 2 for Cr; if not informed, Y is used.'''
        if not (-1 <= component <= 2):
            print('''Component value must be - 1, if you want to calculate the metric
                for all channels or between 0 and 2, corresponding to the wanted channel.''')
            return
        multiple_components = False
        if len(test_array[0][0][0][0]) > 1:
            multiple_components = True
        for t in range(len(original_array)):
            for f in range(len(original_array[0])):
                # print(original_array[t][f].shape)MSSIM
                if multiple_components and component > -1:
                    ori_arr = np.split(original_array[t][f], 3, axis=2)
                    test_arr = np.split(test_array[t][f], 3, axis=2)
                    if np.array_equal(ori_arr[component], test_arr[component]):
                        print(f'PSNR [tuple {t}, frame {f}]: inf\n')
                    else:
                        psnr = skimage.metrics.peak_signal_noise_ratio(
                            ori_arr[component], test_arr[component])
                        print(f'PSNR [tuple {t}, frame {f}]: {psnr}\n')
                else:
                    if np.array_equal(original_array[t][f], test_array[t][f]):
                        print(f'PSNR [tuple {t}, frame {f}]: inf\n')
                    else:
                        psnr = skimage.metrics.peak_signal_noise_ratio(
                            original_array[t][f], test_array[t][f])
                        if component == -1:
                            psnr = psnr/3
                            # maybe change the treatment for multiple channels later
                            # weighted avarage?
                        print(f'PSNR [tuple {t}, frame {f}]: {psnr}\n')

    @staticmethod
    def get_mssim(original_array, test_array, component=0):
        '''Calculates MSSIM metric for each frame in original and test arrays 
        and returns it. If component is -1, it will return MSSIM for all channels; 
        otherwise, inform 0 for Y, 1 for Cb and 2 for Cr; if not informed, Y is used.'''
        if not (-1 <= component <= 2):
            print('''Component value must be - 1, if you want to calculate the metric
                for all channels or between 0 and 2, corresponding to the wanted channel.''')
            return
        multiple_components = False
        if len(test_array[0][0][0][0]) > 1:
            multiple_components = True
        for t in range(len(original_array)):
            for f in range(len(original_array[0])):
                # print(original_array[t][f].shape)
                if multiple_components and component > -1:
                    ori_arr = np.split(original_array[t][f], 3, axis=2)
                    test_arr = np.split(test_array[t][f], 3, axis=2)
                    mssim = skimage.metrics.structural_similarity(
                        ori_arr[component], test_arr[component], channel_axis=2)
                    print(f'MSSIM [tuple {t}, frame {f}]: {mssim}\n')
                else:
                    print(original_array[t][f].shape)
                    mssim = skimage.metrics.structural_similarity(
                        original_array[t][f], test_array[t][f], channel_axis=2)
                    print(f'MSSIM [tuple {t}, frame {f}]: {mssim}\n')

    @staticmethod
    def get_tuple(path: str, number: str):
        dir = os.path.join(path, '{:04d}'.format(number))
        try:
            im1 = Image.open(os.path.join(dir, 'im1.png'))
            im2 = Image.open(os.path.join(dir, 'im2.png'))
            im3 = Image.open(os.path.join(dir, 'im3.png'))
            # im1.show()
            return (im1, im2, im3)
        except Exception as e:
            print('Something went wrong!')
            print(e)

    @staticmethod
    def generate_ndarray_from_video(component: int, video, sets_of_frames):
        '''Generates a ndarray using the given component, frames, and VideoParser object; component: inform 0 for Y,
        1 for Cb and 2 for Cr; video: VideoParser object; frames: Tuple[Tuple[int]] - inform tuple of testing sets, 
        each one being a tuple containing numbers of frames.'''
        if not (0 <= component <= 2):
            print('Component value must be between 0 and 2.')
            return
        test_list = []
        height = video.height
        width = video.width
        for set in sets_of_frames:
            aux = []
            for data in set:
                frame = video.get_frame(data)[component]
                frame = np.reshape(frame, (height, width, 1))
                aux.append(frame)
            test_list.append(aux)
        ndarray = np.array(test_list)
        return ndarray

    @staticmethod
    def generate_ndarray_from_tuple(frames_tuple: Tuple):
        frames_list = []
        height = frames_tuple[0][0].height
        width = frames_tuple[0][0].width
        print(height, ' ', width)
        for set in frames_tuple:
            aux = []
            for im in set:
                array = np.array(im)
                print(array.shape)
                aux.append(array)
            frames_list.append(aux)
        ndarray = np.array(frames_list)
        print(ndarray.shape)
        # print(len(ndarray[0]))
        # print(ndarray[1])
        # print(ndarray)
        return ndarray


'''
parser = VideoParser("ice_cif.y4m")
y, cb, cr = parser.get_frame(40)
parser.plot_component_frame(2)
parser.plot_component_frame(2, sample=2)
parser.plot_frame(2)
f1 = parser.frame_to_image(parser.get_frame(2))
f2 = parser.frame_to_image(parser.get_frame(2))
f1_narr = parser.get_frame(3)[0]
f2_narr = parser.get_frame(2)[0]
print(f1_narr)

arr = VideoParser.generate_ndarray(0, ((2, parser), (3, parser)))
VideoParser.get_psnr(arr)
VideoParser.get_mssim(arr)'''

bus = VideoParser('bus_cif.y4m')
bus_mci = VideoParser('bus_cif_recMCI.y4m')
bus_vsbmc = VideoParser('bus_cif_recVSBMC.y4m')

ori_arr = VideoParser.generate_ndarray_from_video(
    0, bus, ((1, 2, 4), (3, 6, 8)))
test_arr = VideoParser.generate_ndarray_from_video(
    0, bus_mci, ((1, 2, 4), (3, 6, 8)))
test_arr2 = VideoParser.generate_ndarray_from_video(
    0, bus_vsbmc, ((1, 2, 4), (3, 6, 8)))

VideoParser.get_psnr(ori_arr, test_arr)
VideoParser.get_mssim(ori_arr, test_arr2)

t1 = VideoParser.get_tuple(
    '/home/gabrielafs/Documentos/UFSC/Video_coding/datasets/target/00001', 389)
t2 = VideoParser.get_tuple(
    '/home/gabrielafs/Documentos/UFSC/Video_coding/datasets/target/00001', 402)

array = VideoParser.generate_ndarray_from_tuple((t1, t2))
array_modified = array*2

VideoParser.get_mssim(array, array_modified, -1)
VideoParser.get_psnr(array, array_modified, -1)

'''
im = Image.open(
    '/home/gabriela/Documents/videos_raw/Middlebury/Art-2views-3illum-3exp/Art/Illum1/Exp1/view1.png')
im2 = Image.open(
    '/home/gabriela/Documents/videos_raw/Middlebury/Art-2views-3illum-3exp/Art/Illum1/Exp1/view5.png')


a = VideoParser.generate_ndarray_from_tuple(0, (im, im2))
VideoParser.get_psnr(a, a)

'''

'''
test_list = []
for set in ((1, 2, 3),):
    aux = []
    for data in set:
        frame = bus.get_frame(data)[0]
        frame = np.reshape(frame, (bus.height, bus.width, 1))
        aux.append(frame)
    test_list.append(aux)


ndarray = np.array(test_list)
print(len(ndarray[0][0][0][0]))
print(array.shape)
print(array[0][0].size)
print(len(array[0][0]))
aaa = np.split(array[0][0], 3, axis=2)
print(len(aaa))
a = np.reshape(aaa[0], (256, 448))
image = Image.fromarray(a)
image.show()
'''
