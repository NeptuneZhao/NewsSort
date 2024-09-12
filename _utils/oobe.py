# Important

ABSOLUTE_PATH = r'D:\Repositories\NewsSort\_utils'

def UseGPU() -> bool:
    Y = input('是否使用GPU？(y/Y): ')
    return True if (Y == 'y' or Y == 'Y') else False

def FileConv(path: str):
    return str(ABSOLUTE_PATH + '\\' + path)