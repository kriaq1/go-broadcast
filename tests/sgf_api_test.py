from src.sgf_api import SGF

if __name__ == '__main__':
    sgf = SGF()
    sgf.add(4, 4, -1)
    sgf.add(16, 16, 1)
    sgf.add(10, 10, -1)
    sgf.save()
