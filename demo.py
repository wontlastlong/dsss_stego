from dsss_stego import *

def get_error_rate(seq_1, seq_2):
	return len(filter(lambda x: not x, seq_1 == seq_2))/float(seq_1.shape[0])

chip_rate = 12000
strength_factor = 17
key = ('lengthy_key'*5)[:32]

img = StegoImage('./test.bmp')
max_bits = img.get_max_bits(chip_rate)
print max_bits

random_data = MoreSecurePRNG('a'*32).get(max_bits)
img.hide(random_data, key, chip_rate, strength_factor)
img.write('./test_red.bmp')


retrieved = img.retrieve(key, chip_rate)
error_rate = get_error_rate(random_data, retrieved)
print error_rate
