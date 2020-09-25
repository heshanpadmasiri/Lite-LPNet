import shutil


def zip_dir(zip_name: str, dir_name: str):
    shutil.make_archive(zip_name, 'zip', dir_name)
