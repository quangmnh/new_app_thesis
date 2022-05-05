from pygame import mixer

a = mixer

a.init()
a.music.load("chim.mp3")
a.music.set_volume(0.7)

a.music.play()
while True:
    print(a.music.get_pos())