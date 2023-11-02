from goprocam import GoProCamera
from goprocam import constants
gopro = GoProCamera.GoPro()
gopro.stream("udp://127.0.0.1:10000",quality='low')
#gopro.stream("udp://192.168.1.102:10000",quality='low')
