from .Identification.data_io import read_conf
from .Identification.speaker_id import main_func
from .TCResnet.TCResnet_KWS import kws_final_func

#options = read_conf()

kws_softmax = Kws_final_func()
spkid_softmax = main_func()