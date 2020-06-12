from src.Controller import *

def create_name(species_name = '', original = False):
    if original:
        while True:
            name = lorem.sentence().replace('.', '')
            if name not in used_names:
                return species_name + ' ' + name
    else:
        return species_name + ' ' + lorem.sentence().replace('.', '')