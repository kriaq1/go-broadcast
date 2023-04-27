from src.sgf_api import SGF

if __name__ == '__main__':
    sgf = SGF()
    sgf.play(4, 4, -1)
    sgf.play(16, 16, 1)
    sgf.play(10, 10, -1)
    sgf.save()
