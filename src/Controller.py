from Game.game import Game
import random
import numpy as np 
import tensorflow as tf
import copy
import time
import shaper
from os import system, name 
from time import sleep 
from lorem.text import TextLorem
import sys
import json
import codecs  
import gc

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

gc.enable()
#gc.set_debug(gc.DEBUG_LEAK)

lorem = TextLorem(srange=(1,2))
used_names = []