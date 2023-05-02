from src.sgf_api import SGFWriter

if __name__ == '__main__':
    sgf = SGFWriter()
    sgf.add(4, 4, -1)
    sgf.add(20, 20, 1)
    sgf.add(1, 1, -1)
    sgf.add(16, 16, 1)
    sgf.add(10, 11, -1)
    sgf.add(2, 2, -1)
    sgf.change_move(0, 3, 3)
    print(sgf.is_legal(4, 4, 1))
    print(sgf.get_sequence())
    sgf.save()
    sgf.save('sgf/2.sgf')
