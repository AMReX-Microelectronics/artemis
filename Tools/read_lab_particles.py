import numpy as np
from glob import glob
import os

it = 1
fn = "./lab_frame_data/" + 'snapshot' + str(it).zfill(5) + "/particle1/"

print(fn)

def get_particle_field(field):
    files = glob(os.path.join(fn, field + '_*'))
    all_data = np.array([])
    for f in files:
        data = np.fromfile(f)
        all_data = np.concatenate((all_data, data))
    return all_data
    
x = get_particle_field('x')
z = get_particle_field('z')

ux = get_particle_field('ux')
uz = get_particle_field('uz')