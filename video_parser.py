from copy import copy
from typing import Tuple
import numpy as np
import matplotlib.pyplot as matp
from PIL import Image
import scipy.ndimage
import skimage


class VideoParser():
    def __init__(self, file_name, height = 0, width = 0, fps = '', y_size = 0, cr_size = 0, cb_size = 0) -> None:
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
                w = a[a.find(ord('W'))+1 :]
                w = w[:w.find(ord(' '))]
                self.__width = int(w)
                h = a[a.find(ord('H'))+1 :]
                h = h[:h.find(ord(' '))]
                self.__height = int(h)
                fps = a[a.find(ord('F'))+1 :]
                fps = fps[:fps.find(ord(' '))]
                self.__fps = fps
                print(self.__height, self.__width, fps)
                end = f.tell()

                self.__size[0] = self.__height*self.__width
                f.seek(6+self.__size[0],1)
                self.__size[1] = int((len(f.readline())-6)/2)
                self.__size[2] = self.__size[1]
                print(self.__size)
                return end
                
        except Exception as e:
            print(f'Erro: {e}')
    
    def get_frame(self, num_frame:int):
        '''input: file name, frame's number; returns a tuple of numpy.array video samples'''
        dtype = np.dtype('B')
        if num_frame < 1:
            print('Frame number must be greater than 0')
            return
        try:
            with open(self.__file_name, 'rb') as f:
                f.seek(self.__end_header)
                f.seek(6*num_frame + int((num_frame-1)*self.__height*self.__width*1.5), 1)
                sample_y = np.frombuffer(f.read(int(self.__height*self.__width)), dtype=dtype)
                sample_cb = np.frombuffer(f.read(int(self.__height*self.__width/4)), dtype=dtype)
                sample_cr = np.frombuffer(f.read(int(self.__height*self.__width/4)), dtype=dtype)
                return sample_y, sample_cb, sample_cr
               
        except Exception as e:
            print(f'Error: {e}')

    def plot_component_frame(self, num_frame, filename=' ', sample=0):
        '''Extracts the chosen frame from the raw file and creates a .pdf in grayscale using 
        the given sample and filename. Sample: inform 0 for Y, 1 for Cb and 2 for Cr; if not informed, Y is used.'''
        if not(0 <= sample <= 2):
            print('Sample value must be between 0 and 2.')
        component_sample = self.get_frame(num_frame)[sample]
        print(self.__size[0], self.__size[sample])
        print((self.__size[0]/self.__size[sample]))
        print(self.__height)
        if sample > 0:
            im = self.component_to_image(component_sample, (self.__size[0]/self.__size[sample])/2)
        else:
            im = self.component_to_image(component_sample)
        matp.figure()
        if filename == ' ':
            filename = self.__file_name[:len(self.__file_name)-4] + '_' + str(num_frame) + '_' + str(sample)
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
                filename = self.__file_name[:len(self.__file_name)-4] + '_' + str(num_frame)
            im.save(f'{filename}.png')
        except Exception as e:
            print('Something went wrong!')
            print(e)
    
    def component_to_image(self, component_sample, sampling=1):
        component_sample = np.reshape(component_sample, (int(self.__height/sampling), int(self.__width/sampling)))
        im = Image.fromarray(np.uint8(component_sample), "L")
        return im

    def frame_to_image(self, frame):
        y, cb, cr = frame
        ups_cb, ups_cr = cb, cr
        if (self.__size[0]/self.__size[1]) == 4:
            y = np.reshape(y, (int(self.__height), int(self.__width)))
            cb = np.reshape(cb, (int(self.__height/2), int(self.__width/2)))
            cr = np.reshape(cr, (int(self.__height/2), int(self.__width/2)))
            ups_cb = np.kron(cb, np.ones((2,2), dtype=np.dtype('B')))
            ups_cr = np.kron(cr, np.ones((2,2), dtype=np.dtype('B')))
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


parser = VideoParser("ice_cif.y4m")
y, cb, cr = parser.get_frame(40)
parser.plot_component_frame(2)
parser.plot_component_frame(2, sample=2)
parser.plot_frame(2)
f1 = parser.frame_to_image(parser.get_frame(2))
f2 = parser.frame_to_image(parser.get_frame(2))
f1_narr = parser.get_frame(2)[0]
f2_narr = parser.get_frame(2)[0]
print(f1_narr)
f2_narr = copy(f2_narr)
f2_narr[0] = 162


print(skimage.metrics.peak_signal_noise_ratio(f1_narr, f2_narr))