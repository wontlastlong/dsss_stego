import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from skcuda.fft import fft, Plan, ifft
import wave
from struct import unpack, pack
from Crypto.Util.number import bytes_to_long, long_to_bytes
from Crypto.Cipher import AES
import sys
from PIL import Image
import ffmpeg
import os
import shutil

def bytes_to_bits(data):
	data_bit_sequence = np.empty(len(data)*8, np.int8)
	bit_pos = 0
	for v in data:
		current_byte = ord(v)
		for j in range(8):
			data_bit_sequence[bit_pos] = 1 if ((current_byte & 1) == 1) else -1
			current_byte >>= 1
			bit_pos += 1

	return data_bit_sequence

def bits_to_bytes(data_bit_sequence):
	data = ''
	current_byte = 0
	bits_retrieved = 0
	for v in data_bit_sequence:
		if v == 1:
			current_byte = (current_byte >> 1) | 0b10000000
		else:
			current_byte >>= 1
		bits_retrieved += 1

		if bits_retrieved == 8:
			data += chr(current_byte)
			bits_retrieved = 0
			current_byte = 0

	if bits_retrieved != 0:
		data += chr(current_byte >> (8 - bits_retrieved))

	return data


class StegoObject:
	def __init__(self, cover, max_cell_value):
		self.cover = cover
		self.max_cell_value = max_cell_value

	def hide(self, data_bit_sequence, pn_gen, chip_rate, strength_factor):
		cover_size = self.cover.shape[0]
		cover_fft_size = cover_size/2 + 1
		max_data_bits = np.floor(cover_fft_size/chip_rate).astype(np.int32)
		bit_data_len = data_bit_sequence.shape[0]

		if bit_data_len > max_data_bits:
			raise Exception('Length of payload exceeds capacity of cover object (Maximum length: %d bits, %d bytes)' % 
				(max_data_bits, max_data_bits // 8))

		'''
		data_bit_sequence = np.empty(bit_data_len, np.int8)
		for i in range(len(data)):
			current = ord(data[i])
			for j in range(8):
				data_bit_sequence[i*8 + j] = 1 if ((current & 1) == 1) else -1
				current >>= 1
				
			#print
		'''
		cover_reduced = gpuarray.to_gpu(self.cover/self.max_cell_value)
		cover_reduced_fft = gpuarray.empty(cover_fft_size, np.complex64)

		plan_direct = Plan(cover_size, np.float32, np.complex64)
		fft(cover_reduced, cover_reduced_fft, plan_direct)

		data_spread = np.zeros(cover_fft_size, dtype=np.float32)

		for i in range(bit_data_len):
			for j in range(chip_rate):
				data_spread[i*chip_rate + j] = data_bit_sequence[i]

		pn = pn_gen.get(cover_fft_size)
	
		data_modulated = strength_factor*pn*data_spread

		cover_changed_fft = gpuarray.to_gpu(cover_reduced_fft.get() + data_modulated)
		cover_changed = gpuarray.empty(cover_size, np.float32)

		plan_inverse = Plan(cover_size, np.complex64, np.float32)
		ifft(cover_changed_fft, cover_changed, plan_inverse, True)
		self.cover = cover_changed.get()

		for i in range(cover_size):
			if self.cover[i] > 1.0:
				self.cover[i] = 1.0
			elif self.cover[i] < -1.0:
				self.cover[i] = -1.0

		self.cover = self.cover*self.max_cell_value

	def retrieve(self, pn_gen, chip_rate):
		cover_size = self.cover.shape[0]
		cover_fft_size = cover_size/2 + 1
		max_data_bits = np.floor(cover_fft_size/chip_rate).astype(np.int32)

		cover_reduced = gpuarray.to_gpu(self.cover/self.max_cell_value)
		cover_reduced_fft = gpuarray.empty(cover_fft_size, np.complex64)

		plan_direct = Plan(cover_size, np.float32, np.complex64)
		fft(cover_reduced, cover_reduced_fft, plan_direct)

		cover_reduced_fft = cover_reduced_fft.get()

		pn = pn_gen.get(cover_fft_size)
		cover_reduced_fft *= pn		

		data_bit_sequence = np.empty(max_data_bits, np.int8)

		for i in range(max_data_bits):
			chip_sum = np.sum(cover_reduced_fft[i*chip_rate:(i + 1)*chip_rate])

			data_bit_sequence[i] = 1 if chip_sum > 0.0 else -1
					
		return data_bit_sequence

# for tests
class InsecurePRNG:
	def __init__(self, key):
		np.random.seed(bytes_to_long(key) % 2**32)
		self.choice_bits = np.asarray([-1.0, 1.0], np.float32)

	def reinit(self, key):
		np.random.seed(bytes_to_long(key) % 2**32)

	def get(self, nbits):
		return np.random.choice(self.choice_bits, nbits)

class MoreSecurePRNG:
	def __init__(self, key):
		self.block_size = 16*8
		self.reinit(key)

	def reinit(self, key):
		self.cipher = AES.new(key, AES.MODE_ECB)
		self.counter = 0
		self.stored = np.empty(0, dtype=np.float32)

	def get(self, nbits):
		bit_sequence = np.empty(nbits, dtype=np.float32)

		left = nbits
		bit_pos = 0
		if self.stored.shape[0] != 0:
			if self.stored.shape[0] >= nbits:
				bit_sequence = self.stored[:nbits]
				self.stored = self.stored[nbits:]
				return bit_sequence
			else:
				bit_sequence[:self.stored.shape[0]] = self.stored
				left -= self.stored.shape[0]
				bit_pos += self.stored.shape[0]

		while left >= self.block_size:
			counter_bytes = long_to_bytes(self.counter)
			counter_bytes += '\x00'*(16 - len(counter_bytes))
			generated = bytes_to_bits(self.cipher.encrypt(counter_bytes))
			bit_sequence[ bit_pos : bit_pos + self.block_size ] = generated
			self.counter += 1
			bit_pos += self.block_size
			left -= self.block_size

		if left != 0:
			counter_bytes = long_to_bytes(self.counter)
			counter_bytes += '\x00'*(16 - len(counter_bytes))
			generated = bytes_to_bits(self.cipher.encrypt(counter_bytes))
			bit_sequence[ bit_pos : ] = generated[ : left ]
			self.stored = generated[ left : ]
			self.counter += 1

		return bit_sequence
		


class StegoWave16Bit:
	def __init__(self, filename):
		f = wave.open(filename, 'rb')

		self.nframes = f.getnframes()
		self.nchannels = f.getnchannels()
		self.params = f.getparams()

		channels_np = [np.zeros(self.nframes, dtype=np.float32) for i in range(self.nchannels)]

		unpack_format = '<' + 'h'*self.nchannels
		for i in range(self.nframes):
			values = unpack(unpack_format, f.readframes(1))
			for j in range(self.nchannels):
				channels_np[j][i] = values[j]

		self.channels = [StegoObject(cover, 2**15 - 1) for cover in channels_np]
		f.close()

	def write(self, filename):
		f = wave.open(filename, 'wb')
		f.setparams(self.params)

		for i in range(self.nframes):
			frame = ''
			for j in range(self.nchannels):
				frame += pack('<h', int(self.channels[j].cover[i]))
			f.writeframesraw(frame)

		f.close()

	def get_max_bits(self, chip_rate):
		return self.nchannels*np.floor((self.nframes/2 + 1)/chip_rate).astype(np.int32)

	def hide(self, data_bit_sequence, key, chip_rate, strength_factor):
		bit_data_len = data_bit_sequence.shape[0]
		bits_per_channel = np.floor((self.nframes/2 + 1)/chip_rate).astype(np.int32)
		max_data_bits = self.nchannels*bits_per_channel

		if bit_data_len > max_data_bits:
			raise Exception('Data won\'t fit! Maximum data bits: %d bytes: %d' % (max_data_bits, max_data_bits//8))

		pn_gen = MoreSecurePRNG(key)

		data_left = bit_data_len
		channel_id = 0
		bit_data_pos = 0
		while data_left > 0:
			self.channels[channel_id].hide( data_bit_sequence[ bit_data_pos : bit_data_pos + bits_per_channel ], 
				pn_gen, chip_rate, strength_factor )
			data_left -= bits_per_channel
			bit_data_pos += bits_per_channel
			channel_id += 1

		while channel_id != self.nchannels:
			random_bits = pn_gen.get(bits_per_channel)
			self.channels[channel_id].hide( random_bits, pn_gen, chip_rate, strength_factor)
			channel_id += 1

	def retrieve(self, key, chip_rate):
		bits_per_channel = np.floor((self.nframes/2 + 1)/chip_rate).astype(np.int32)
		max_data_bits = self.nchannels*bits_per_channel

		pn_gen = MoreSecurePRNG(key)
		data_bit_sequence = np.empty(max_data_bits, np.int8)

		bit_data_pos = 0
		for channel in self.channels:
			data_bit_sequence[bit_data_pos:bit_data_pos + bits_per_channel] =\
				channel.retrieve(pn_gen, chip_rate)
			bit_data_pos += bits_per_channel

		return data_bit_sequence

class StegoImage:
	def __init__(self, filename):
		f = Image.open(filename).convert('RGB')
		self.w, self.h = f.size
		self.format = f.format
		print self.w, self.h
		pixels = f.load()

		channels_np = [np.zeros(self.w*self.h, dtype=np.float32) for i in range(3)]

		for i in range(self.h):
			for j in range(self.w):
				for k in range(3):
					channels_np[k][i*self.w + j] = pixels[j, i][k] - 127.5

		self.channels = [StegoObject(channel, 127.5) for channel in channels_np]
		f.close()

	def get_max_bits(self, chip_rate):
		return 3*np.floor(((self.w*self.h)/2 + 1)/chip_rate).astype(np.int32)

	def write(self, filename):
		f = Image.new('RGB', (self.w, self.h))
		pixels = f.load()
		for i in range(self.h):
			for j in range(self.w):
				values = [0]*3
				for k in range(3):
					values[k] = (self.channels[k].cover[i*self.w + j] + 127.5).astype(np.int16)

					if values[k] > 255:
						values[k] = 255
					elif values[k] < 0:
						values[k] = 0

				pixels[j, i] = tuple(values)

		f.format = self.format
		f.save(filename)

	def hide(self, data_bit_sequence, key, chip_rate, strength_factor):
		bit_data_len = data_bit_sequence.shape[0]
		bits_per_channel = np.floor(((self.w*self.h)/2 + 1)/chip_rate).astype(np.int32)
		max_data_bits = 3*bits_per_channel

		if bit_data_len > max_data_bits:
			raise Exception('Data won\'t fit! Maximum data bits: %d bytes: %d' % (max_data_bits, max_data_bits//8))

		pn_gen = MoreSecurePRNG(key)

		data_left = bit_data_len
		channel_id = 0
		bit_data_pos = 0
		while data_left > 0:
			self.channels[channel_id].hide( data_bit_sequence[ bit_data_pos : bit_data_pos + bits_per_channel ], 
				pn_gen, chip_rate, strength_factor )
			data_left -= bits_per_channel
			bit_data_pos += bits_per_channel
			channel_id += 1

		while channel_id != 3:
			random_bits = pn_gen.get(bits_per_channel)
			self.channels[channel_id].hide( random_bits, pn_gen, chip_rate, strength_factor)
			channel_id += 1

	def retrieve(self, key, chip_rate):
		bits_per_channel = np.floor(((self.w*self.h)/2 + 1)/chip_rate).astype(np.int32)
		max_data_bits = 3*bits_per_channel

		pn_gen = MoreSecurePRNG(key)
		data_bit_sequence = np.empty(max_data_bits, np.int8)

		bit_data_pos = 0
		for channel in self.channels:
			data_bit_sequence[bit_data_pos:bit_data_pos + bits_per_channel] =\
				channel.retrieve(pn_gen, chip_rate)
			bit_data_pos += bits_per_channel

		return data_bit_sequence

'''
def StegoGIF:
	def __init__(self, filename):
		dirname = filename + '_frames_original'
		if os.path.exists(dirname):
			if not os.path.isdir(dirname):
				raise Exception('Cannot create directory for extracted frames! (name conflict)')
		else:
			os.mkdir(dirname)

		stream = ffmpeg.input(filename)
		stream = ffmpeg.output(stream, dirname + '/frame%06d.bmp')
		ffmpeg.run(stream)
		
		frames = sorted(os.listdir(dirname))
		self.covers = [StegoImage(frame) for frame in frames]
		shutil.rmdir(dirname)

	def write(self, filename):
		dirname = filename + '_frames_stegoed'
		if os.path.exists(dirname):
			if not os.path.isdir(dirname):
				raise Exception('Cannot create directory for extracted frames! (name conflict)')
		else:
			os.mkdir(dirname)

		for i, cover in enumerate(self.covers):
			cover.write(dirname + '/frame%06d.bmp' % (i + 1))

		stream = ffmpeg.input(dirname + '/frame%06d.bmp')
		stream = ffmpeg.output(stream, filename)
		ffmpeg.run(stream)
		shutil.rmdir(dirname)
'''
'''
	def hide(self, data, key, chip_rate, strength_factor):
		bits_per_channel = np.floor(((self.w*self.h)/2 + 1)/chip_rate).astype(np.int32)
		max_data_bits = 3*bits_per_channel
'''



def get_inequals(a, b):
	if a == b:
		return 0
	else:
		return 1