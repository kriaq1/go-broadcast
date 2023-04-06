from src.manual_editor import ManualEditor


def manual_editor_test():
    editor = ManualEditor()
    editor.put_stone(0, 0, 1)
    editor.put_stone(3, 3, -1)
    editor.put_stone(4, 3, 1)
    editor.put_stone(3, 4, -1)
    editor.put_stone(6, 6, 1)
    editor.put_stone(9, 5, -1)
    editor.put_stone(5, 3, 1)
    editor.put_stone(5, 3, -1)
    editor.put_stone(4, 4, 1)
    editor.put_stone(5, 4, -1)
    editor.put_stone(2, 6, 1)
    editor.put_stone(3, 6, -1)
    editor.put_stone(6, 3, 1)
    editor.put_stone(1, 3, -1)
    editor.put_stone(2, 0, 1)
    editor.remove_stone(9, 5)
    editor.remove_stone(10, 10)
    editor.put_stone(9, 6, 1)
    editor.put_stone(8, 6, -1)
    editor.put_stone(13, 5, 1)
    editor.put_stone(4, 16, -1)
    editor.put_stone(9, 12, 1)
    editor.put_stone(2, 11, -1)
    editor.put_stone(17, 4, 1)
    return editor.get()


if __name__ == '__main__':
    manual_editor_test().print_to_console()
