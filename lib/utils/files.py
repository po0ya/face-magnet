import os
import shutil


def make_backup_delete(file_path, postfix='.backup'):

    if os.path.exists(file_path):
        backup_path = file_path+postfix
        shutil.copyfile(file_path,backup_path)
        os.remove(file_path)
        print('File {:s} is backed up to {:s} and removed'.format(file_path,backup_path))
